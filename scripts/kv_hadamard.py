#!/usr/bin/env python3
"""Test C (PREREG amendment 1): fixed H512 before the two-level NVFP4 MLA latent record.

Stand-in attention input for MLA layer A from the routed-block capture of layer C (C = A-1
for the decisive set, C = A for the reported-only sensitivity runs):

    a = gamma_in(A) * (m_C / gamma_post(C))          (BF16)
    c = kv_a_layernorm(kv_a_proj_with_mqa(a))        (BF16, 512 dims, no RoPE payload)
    q_abs[h] = W_UK[h]^T q_b_proj(q_a_layernorm(q_a_proj(a)))[h]

Record mirror = b12x/attention/_shared/mla/kv_cache.py, per_token_scale=True:
    s_t = token_amax / (6*448);  byte = e4m3_rn_satfinite(group_amax / s_t / 6)
    value = e2m1_rne_satfinite(v / (dec(byte) * s_t));  read = value * dec(byte) * s_t
Arms: U (unrotated, production), H (normalized Sylvester H512, decisive), HD (H512 with
fixed random signs, reported only). Rotated arms quantize H c and invert after dequant, so
every error is measured in the original latent space.

Per window: latent error energy over all 2,048 keys, and attention-logit error energy over
the 48 Test-B query positions x 64 heads x causal keys.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

CAMPAIGN = Path("<workspace>/trellismx-review-20260922/glm53-hadamard-shapleymcg-kld")
sys.path.insert(0, str(CAMPAIGN))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from glm53_nvfp4.p8_coupled_scale import block_hadamard  # noqa: E402
from remainder_damage import wait_for_production_idle  # noqa: E402

CARRIER = Path("<home>/models/GLM-5.3-Flash-NVFP4")
CAPTURE = Path("<data-volume>/p8-remainder-kvrot-20260923/capture/layers")
FIT = json.loads((Path(__file__).resolve().parents[1] / "fit_ranges.json").read_text())
ROWS, HIDDEN, WINDOW, EPS = 1310720, 4096, 2048, 1e-5
HEADS, NOPE, V, LATENT = 64, 256, 256, 512
QUERIES_PER_WINDOW = 48
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)


class Carrier:
    def __init__(self):
        self.index = json.loads((CARRIER / "model.safetensors.index.json").read_text())["weight_map"]

    def get(self, name: str) -> torch.Tensor:
        with safe_open(str(CARRIER / self.index[name]), framework="pt", device="cpu") as f:
            return f.get_tensor(name)


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
    return (y * weight.float()).to(torch.bfloat16)


def linear_bf16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (x.float() @ w.float().T).to(torch.bfloat16)


def e2m1_rne(x: torch.Tensor) -> torch.Tensor:
    """cvt.rn.satfinite.e2m1: nearest E2M1 level, ties to the even code."""
    levels = E2M1.to(x.device)
    a = x.double().abs().clamp(max=6.0)
    dist = (a[..., None] - levels).abs()
    dist = dist + (torch.arange(8, device=x.device) % 2).double() * 1e-12  # tie -> even code
    q = levels[dist.argmin(dim=-1)]
    return (torch.sign(x.double()) * q).float()


def nvfp4_two_level(c: torch.Tensor) -> torch.Tensor:
    """Quantize + dequantize [N, 512] FP32 latents exactly as the per-token-scale record."""
    g = c.float().reshape(c.shape[0], LATENT // 16, 16)
    group_amax = g.abs().amax(dim=-1)
    token_amax = group_amax.amax(dim=-1, keepdim=True)
    s_t = token_amax * torch.tensor(1.0 / (6.0 * 448.0), dtype=torch.float32)
    inv_latent = torch.where(s_t != 0, 1.0 / s_t, torch.zeros_like(s_t))
    scale_f32 = (group_amax * inv_latent) * torch.tensor(1.0 / 6.0, dtype=torch.float32)
    dec = scale_f32.clamp(max=448.0).to(torch.float8_e4m3fn).float()
    mult = torch.where(dec != 0, (1.0 / dec) * inv_latent, torch.zeros_like(dec))
    q = e2m1_rne(g * mult[..., None])
    return ((q * dec[..., None]) * s_t[..., None]).reshape_as(c)


def reconstruct(c: torch.Tensor, arm: str, signs: torch.Tensor) -> torch.Tensor:
    if arm == "U":
        return nvfp4_two_level(c)
    pre = c * signs if arm == "HD" else c
    rot = block_hadamard(pre, block_size=LATENT)
    back = block_hadamard(nvfp4_two_level(rot), block_size=LATENT)
    return back * signs if arm == "HD" else back


def run_config(carrier: Carrier, attn_layer: int, capture_layer: int, device, handles, signs) -> dict:
    P = "model.language_model.layers"
    A = f"{P}.{attn_layer}.self_attn"
    w = {n: carrier.get(f"{A}.{n}.weight").to(device) for n in
         ("q_a_proj", "q_a_layernorm", "q_b_proj", "kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj")}
    gamma_in = carrier.get(f"{P}.{attn_layer}.input_layernorm.weight").to(device).float()
    gamma_post = carrier.get(f"{P}.{capture_layer}.post_attention_layernorm.weight").to(device).float()
    w_uk = w["kv_b_proj"].float().reshape(HEADS, NOPE + V, LATENT)[:, :NOPE, :]  # [H, 256, 512]
    hid = np.memmap(CAPTURE / f"layer-{capture_layer:03d}/hidden.bf16.bin", dtype="<u2", mode="r", shape=(ROWS, HIDDEN))
    picks = torch.from_numpy(np.linspace(0, WINDOW - 1, QUERIES_PER_WINDOW).round().astype(np.int64)).to(device)
    causal = torch.arange(WINDOW, device=device)[None, :] <= picks[:, None]  # [48, 2048]
    causal = causal[:, None, :].expand(QUERIES_PER_WINDOW, HEADS, WINDOW).reshape(-1, WINDOW)
    arms = ("U", "H", "HD")
    out = {"attn_layer": attn_layer, "capture_layer": capture_layer, "latent_den": [], "logit_den": [],
           "latent_num": {a: [] for a in arms}, "logit_num": {a: [] for a in arms},
           "latent_amax_over_rms": [], "yielded_s": 0.0}
    for window in FIT["fit_window_indices"]:
        if handles:
            out["yielded_s"] += wait_for_production_idle(handles)
        m = torch.from_numpy(np.array(hid[window * WINDOW:(window + 1) * WINDOW], copy=True)).view(torch.bfloat16).to(device)
        if not bool(m.float().abs().amax(dim=-1).gt(0).all()):
            raise RuntimeError(f"unmaterialized capture rows: layer {capture_layer} window {window}")
        a = (gamma_in * (m.float() / gamma_post)).to(torch.bfloat16)
        c = rms_norm(linear_bf16(a, w["kv_a_proj_with_mqa"]), w["kv_a_layernorm"]).float()  # [2048, 512]
        qa = rms_norm(linear_bf16(a[picks], w["q_a_proj"]), w["q_a_layernorm"])
        q = linear_bf16(qa, w["q_b_proj"]).float().reshape(QUERIES_PER_WINDOW, HEADS, NOPE)
        q_abs = torch.einsum("thd,hdc->thc", q, w_uk).reshape(-1, LATENT)  # [48*64, 512]
        logits = q_abs @ c.T
        out["latent_den"].append(float(c.double().square().sum()))
        out["logit_den"].append(float(logits.double().square()[causal].sum()))
        out["latent_amax_over_rms"].append(float((c.abs().amax(dim=-1) / c.square().mean(dim=-1).sqrt()).mean()))
        for arm in arms:
            err = reconstruct(c, arm, signs) - c
            out["latent_num"][arm].append(float(err.double().square().sum()))
            out["logit_num"][arm].append(float((q_abs @ err.T).double().square()[causal].sum()))
    for metric in ("latent", "logit"):
        den = sum(out[f"{metric}_den"])
        out[f"{metric}_relative"] = {a: sum(out[f"{metric}_num"][a]) / den for a in arms}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True, help="comma list of attn:capture, e.g. 7:6,19:18")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-yield", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    handles = []
    if not args.no_yield:
        import pynvml

        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount()) if i != args.gpu]
    gen = torch.Generator().manual_seed(20260923)
    signs = (torch.randint(0, 2, (LATENT,), generator=gen) * 2 - 1).float().to(device)
    carrier = Carrier()
    results, started = [], time.time()
    with torch.inference_mode():
        for item in args.configs.split(","):
            attn_layer, capture_layer = (int(v) for v in item.split(":"))
            res = run_config(carrier, attn_layer, capture_layer, device, handles, signs)
            results.append(res)
            print(json.dumps({"attn_layer": attn_layer, "capture_layer": capture_layer,
                              "latent_relative": res["latent_relative"], "logit_relative": res["logit_relative"],
                              "yielded_s": res["yielded_s"], "elapsed_s": round(time.time() - started, 1)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "p8-kv-hadamard.v1", "fit_windows": FIT["fit_window_indices"],
                                       "queries_per_window": QUERIES_PER_WINDOW, "results": results,
                                       "protected_roles_opened": []}, indent=1) + "\n")


if __name__ == "__main__":
    main()
