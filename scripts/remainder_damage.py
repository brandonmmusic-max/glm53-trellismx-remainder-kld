#!/usr/bin/env python3
"""Test B (PREREG): layer-level routed-output damage of the two-term FP8 activation carrier.

Method = the campaign's p8_layer_rate_damage: r27 rank sidecars reference-decoded with the
kernel's procedural-MCG E4M3 codebook, executed through the exact coupled reference forward,
damage measured against the BF16 source experts on 64 fit windows x 48 tokens. The only
change is the activation carrier at the two A8 hops:

    a8    : hi = q(x)                          (production)
    a8x2  : hi = q(x), lo = q(x - hi), hi + lo in FP32 (what two FP8 MMAs accumulate)
    exact : x unquantized (FP32)

q = E4M3 with a UE8M0 scale per K32 block (`_qdq_e4m3_k32`, amax policy), the same mirror
the campaign and the 09-04 screen use. The a8/a8 arm is asserted bit-identical to the
campaign's `coupled_expert_reference(quantize_activations=True)`.

GPU etiquette: production keeps serving; before each expert this waits while any other GPU
shows load (production is TP4, so its activity is visible on GPUs this job does not use).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CAMPAIGN = Path("<workspace>/trellismx-review-20260922/glm53-hadamard-shapleymcg-kld")
sys.path.insert(0, str(CAMPAIGN))

from glm53_nvfp4.canary_mxfp6_reap import _qdq_e4m3_k32  # noqa: E402
from glm53_nvfp4.capture import LayerCapture  # noqa: E402
from glm53_nvfp4.p8_coupled_scale import (  # noqa: E402
    ACTIVATION_LIMIT,
    COUPLED_SIGN_DRAW,
    HADAMARD_INPUT,
    HADAMARD_RESIDUAL,
    CoupledScaleSet,
    _atom_interleave,
    block_hadamard,
    coupled_expert_reference,
    encode_coupled_scale_weights,
    rotation_signs,
    source_expert_reference,
)
from glm53_nvfp4.p8_layer_rate_damage import (  # noqa: E402
    EXPERTS,
    HIDDEN,
    INTERMEDIATE,
    TOP_K,
    RankLayerStreams,
    decode_expert,
    select_tokens,
    state_codebook,
)
from glm53_nvfp4.shard_index import IndexedCheckpoint, sha256_file  # noqa: E402

SIDECARS = Path("<model-volume>/glm53-trellismx-native6/trellismx-r27-local-checkpoint-v1/sidecars")
SOURCE = Path("<data-volume>/p8-remainder-kvrot-20260923/bf16")
CAPTURE = Path("<data-volume>/p8-remainder-kvrot-20260923/capture")
ROLES = Path("<data-volume>/bmxfp4-glm53/roles/roles-v3.json")

# arm -> (weights, FC1-input carrier, down-input carrier)
ARMS = {
    "P-A8": ("p8", "a8", "a8"),
    "P-A8x2": ("p8", "a8x2", "a8x2"),
    "P-Aexact": ("p8", "exact", "exact"),
    "B-A8": ("bf16", "a8", "a8"),
    "B-A8x2": ("bf16", "a8x2", "a8x2"),
    "B-Aexact": ("bf16", "exact", "exact"),  # closure: coupled mapping of BF16 == source
    # Added after Test A's result, reported only: which hop carries the gain.
    "P-A8x2-fc1only": ("p8", "a8x2", "a8"),
    "P-A8x2-downonly": ("p8", "a8", "a8x2"),
}


def carrier(work: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "exact":
        return work
    hi = _qdq_e4m3_k32(work, 1.0, "amax").float()
    if mode == "a8":
        return hi
    return hi + _qdq_e4m3_k32(work - hi, 1.0, "amax").float()


class DeviceScales:
    def __init__(self, scales: CoupledScaleSet, device: torch.device):
        self.suh = scales.gate_up_suh.to(device).float()
        self.gate_svh = scales.gate_svh.to(device)
        self.up_svh = scales.up_svh.to(device)
        self.down_suh = scales.down_suh.to(device).float()
        self.down_svh = scales.down_svh.to(device).float()
        self.pre_signs = rotation_signs(2 * INTERMEDIATE, draw=COUPLED_SIGN_DRAW, axis=1).to(device)
        self.post_signs = rotation_signs(INTERMEDIATE, draw=COUPLED_SIGN_DRAW, axis=2).to(device)


def input_work(hidden: torch.Tensor, ds: DeviceScales) -> torch.Tensor:
    work = hidden.to(torch.bfloat16).to(torch.float16).float()
    work = block_hadamard(work, block_size=HADAMARD_RESIDUAL)
    work = work * ds.suh
    return block_hadamard(work, block_size=HADAMARD_INPUT)


def coupled_forward(fc1_input: torch.Tensor, weights, ds: DeviceScales, expert: int, mid_mode: str) -> torch.Tensor:
    """coupled_middle_carrier + down + output transform, with a selectable down-input carrier."""
    gate_core, up_core, down_core = weights
    physical_gate = F.linear(fc1_input.float(), gate_core.float()).to(torch.float16).float()
    physical_up = F.linear(fc1_input.float(), up_core.float()).to(torch.float16).float()
    raw = _atom_interleave(physical_gate, physical_up)
    pair_scales = _atom_interleave(ds.gate_svh[expert:expert + 1], ds.up_svh[expert:expert + 1])[0].float()
    pre = block_hadamard(raw, block_size=HADAMARD_INPUT)
    pre = block_hadamard(pre * pair_scales, block_size=HADAMARD_INPUT)
    pre = pre * ds.pre_signs
    gate = pre[:, 0::2].clamp(max=ACTIVATION_LIMIT)
    up = pre[:, 1::2].clamp(-ACTIVATION_LIMIT, ACTIVATION_LIMIT)
    middle = F.silu(gate) * up
    middle = block_hadamard(middle * ds.post_signs, block_size=HADAMARD_INPUT)
    middle = middle * ds.down_suh[expert]
    middle = block_hadamard(middle, block_size=HADAMARD_INPUT)
    down_input = carrier(middle, mid_mode)
    down = F.linear(down_input.float(), down_core.float()).to(torch.float16).float()
    route = block_hadamard(down, block_size=HADAMARD_INPUT) * ds.down_svh
    return block_hadamard(route, block_size=HADAMARD_RESIDUAL)


def sidecar_scales(layer: int) -> tuple[CoupledScaleSet, list[Path]]:
    from safetensors import safe_open

    paths = [SIDECARS / f"p8-layer-{layer:03d}-tp4-rank-{r}.safetensors" for r in range(4)]
    suh, svh, parts = None, None, []
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as src:
            s, d, inter = (src.get_tensor(n) for n in ("gate_up_suh_fp16", "down_svh_fp16", "intermediate_scales_fp16"))
        if suh is None:
            suh, svh = s, d
        elif not (torch.equal(suh, s) and torch.equal(svh, d)):
            raise RuntimeError(f"shared suh/svh differ across ranks: {path}")
        local = inter.shape[1] // 3
        parts.append((inter[:, :local], inter[:, local:2 * local], inter[:, 2 * local:]))
    scales = CoupledScaleSet(
        gate_up_suh=suh.contiguous(), down_svh=svh.contiguous(),
        gate_svh=torch.cat([p[0] for p in parts], dim=1).contiguous(),
        up_svh=torch.cat([p[1] for p in parts], dim=1).contiguous(),
        down_suh=torch.cat([p[2] for p in parts], dim=1).contiguous(),
        source_path=paths[0], source_sha256="", source_metadata={}, tensor_hashes={},
    )
    scales.validate()
    return scales, paths


def wait_for_production_idle(handles, threshold: int = 5) -> float:
    import pynvml

    waited = 0.0
    while any(pynvml.nvmlDeviceGetUtilizationRates(h).gpu > threshold for h in handles):
        time.sleep(2.0)
        waited += 2.0
    return waited


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tokens-per-window", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-yield", action="store_true")
    parser.add_argument("--max-experts", type=int, default=EXPERTS, help="smoke tests only")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    handles = []
    if not args.no_yield:
        import pynvml

        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount()) if i != args.gpu]

    source = IndexedCheckpoint(SOURCE, SOURCE / "model.safetensors.index.json")
    capture = LayerCapture(CAPTURE, args.layer, ROLES, max_samples=16, data_role="fit",
                           sampling_strategy="domain-balanced")
    tokens = select_tokens(capture, args.tokens_per_window)
    scales, rank_paths = sidecar_scales(args.layer)
    streams = RankLayerStreams(rank_paths)
    bits = streams.bits
    codebook = state_codebook(bits, device)
    ds = DeviceScales(scales, device)

    hidden = tokens["hidden"].to(device)
    ids = tokens["ids"].to(device)
    weights = tokens["weights"].to(device).float()
    T = hidden.shape[0]
    window_of_token = torch.arange(T, device=device) // args.tokens_per_window
    S = {arm: torch.zeros(T, HIDDEN, dtype=torch.float64, device=device) for arm in ARMS}
    R = torch.zeros(T, HIDDEN, dtype=torch.float64, device=device)
    prefix = source.expert_prefix(args.layer, 0).split(f"layers.{args.layer}.")[0]
    nmse = {"gate": [], "up": [], "down": []}
    parity_checked = False
    yielded, started = 0.0, time.time()
    with torch.inference_mode():
        for expert in range(args.max_experts):
            if handles:
                yielded += wait_for_production_idle(handles)
            token_idx, slot_idx = torch.nonzero(ids == expert, as_tuple=True)
            base_name = f"{prefix}layers.{args.layer}.mlp.experts.{expert}"
            original = [source.get(f"{base_name}.{p}.weight").to(device).float()
                        for p in ("gate_proj", "up_proj", "down_proj")]
            target = encode_coupled_scale_weights(*original, scales, expert=expert, intermediate_draw=COUPLED_SIGN_DRAW)
            decoded = decode_expert(streams.expert(expert), bits, codebook, device)
            for name, tgt, dec in zip(("gate", "up", "down"), target, decoded):
                nmse[name].append(float(((dec - tgt).double().square().sum() / tgt.double().square().sum()).item()))
            if token_idx.numel():
                x = hidden[token_idx]
                w = weights[token_idx, slot_idx][:, None].double()
                base = source_expert_reference(x, *original)
                R.index_add_(0, token_idx, w * base.double())
                work = input_work(x, ds)
                fc1_in = {mode: carrier(work, mode) for mode in ("a8", "a8x2", "exact")}
                for arm, (wsrc, in_mode, mid_mode) in ARMS.items():
                    wts = decoded if wsrc == "p8" else target
                    out = coupled_forward(fc1_in[in_mode], wts, ds, expert, mid_mode)
                    if arm == "P-A8" and not parity_checked:
                        ref = coupled_expert_reference(x, decoded, scales, expert=expert,
                                                       intermediate_draw=COUPLED_SIGN_DRAW, quantize_activations=True)
                        if not torch.equal(out, ref):
                            raise RuntimeError("a8/a8 arm is not bit-identical to coupled_expert_reference")
                        parity_checked = True
                    S[arm].index_add_(0, token_idx, w * (out - base).double())
            del original, target, decoded
            if expert % 32 == 31:
                print(json.dumps({"layer": args.layer, "expert": expert + 1, "elapsed_s": round(time.time() - started, 1),
                                  "yielded_s": yielded}), flush=True)
    if not parity_checked:
        raise RuntimeError("parity check never ran")
    windows = len(capture.window_indices)
    out_norm = R.square().sum(dim=-1)
    out_w = torch.zeros(windows, dtype=torch.float64, device=device).index_add_(0, window_of_token, out_norm)
    domains = [capture.window_domains[w] for w in capture.window_indices]
    arms = {}
    for arm, s in S.items():
        per_token = s.square().sum(dim=-1)
        per_window = torch.zeros(windows, dtype=torch.float64, device=device).index_add_(0, window_of_token, per_token)
        arms[arm] = {"damage_sum": float(per_token.sum()), "relative_to_routed_output": float(per_token.sum() / out_norm.sum()),
                     "per_window_damage": per_window.tolist()}
    payload = {
        "schema": "p8-remainder-layer-damage.v1", "layer": args.layer, "bits": bits, "tokens": int(T),
        "tokens_per_window": args.tokens_per_window, "fit_windows": [int(w) for w in capture.window_indices],
        "window_domains": domains, "per_window_routed_output_norm": out_w.tolist(), "arms": arms,
        "arm_definitions": {k: {"weights": v[0], "fc1_input": v[1], "down_input": v[2]} for k, v in ARMS.items()},
        "weight_nmse_mean": {k: float(np.mean(v)) for k, v in nmse.items()},
        "weight_nmse_max": {k: float(np.max(v)) for k, v in nmse.items()},
        "a8_parity_with_coupled_expert_reference": "bit-identical",
        "sidecars": [{"path": str(p), "sha256": sha256_file(p)} for p in rank_paths],
        "source_index_sha256": sha256_file(SOURCE / "model.safetensors.index.json"),
        "capture_manifest_sha256": sha256_file(CAPTURE / "capture-manifest.json"),
        "roles_sha256": sha256_file(ROLES), "elapsed_s": time.time() - started, "yielded_to_production_s": yielded,
        "protected_roles_opened": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + "\n")
    rel = {k: v["relative_to_routed_output"] for k, v in arms.items()}
    print(json.dumps({"layer": args.layer, "bits": bits, "relative": rel, "weight_nmse_mean": payload["weight_nmse_mean"],
                      "elapsed_s": round(payload["elapsed_s"], 1)}), flush=True)


if __name__ == "__main__":
    main()
