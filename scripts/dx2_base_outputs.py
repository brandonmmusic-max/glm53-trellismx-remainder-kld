#!/usr/bin/env python3
"""K2 input: the Test B fit tokens and their BF16-source routed outputs, for the in-image closure."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remainder_damage import (CAPTURE, EXPERTS, HIDDEN, ROLES, SOURCE, IndexedCheckpoint, LayerCapture,  # noqa: E402
                              select_tokens, source_expert_reference)

for layer in (int(v) for v in sys.argv[1:]):
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda:0")
    source = IndexedCheckpoint(SOURCE, SOURCE / "model.safetensors.index.json")
    capture = LayerCapture(CAPTURE, layer, ROLES, max_samples=16, data_role="fit", sampling_strategy="domain-balanced")
    tokens = select_tokens(capture, 48)
    hidden, ids, weights = tokens["hidden"].to(dev), tokens["ids"].to(dev), tokens["weights"].to(dev).float()
    base = torch.zeros(hidden.shape[0], HIDDEN, dtype=torch.float64, device=dev)
    prefix = source.expert_prefix(layer, 0).split(f"layers.{layer}.")[0]
    with torch.inference_mode():
        for e in range(EXPERTS):
            t_idx, s_idx = torch.nonzero(ids == e, as_tuple=True)
            if not t_idx.numel():
                continue
            g, u, d = (source.get(f"{prefix}layers.{layer}.mlp.experts.{e}.{p}.weight").to(dev).float()
                       for p in ("gate_proj", "up_proj", "down_proj"))
            base.index_add_(0, t_idx, weights[t_idx, s_idx][:, None].double() * source_expert_reference(hidden[t_idx], g, u, d).double())
    torch.save({"layer": layer, "rows": torch.from_numpy(tokens["rows"]), "hidden": tokens["hidden"],
                "ids": tokens["ids"].to(torch.int32), "weights": tokens["weights"], "base": base.float().cpu(),
                "tokens_per_window": 48}, Path(__file__).resolve().parents[1] / f"results/k2/base-layer-{layer:03d}.pt")
    print("layer", layer, "tokens", hidden.shape[0], "base rms", float(base.square().mean().sqrt()), flush=True)
