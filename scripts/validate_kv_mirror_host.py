#!/usr/bin/env python3
"""Dump real stand-in latents (layer 19 from capture 18, two fit windows; plus their H512
rotation) for the in-image writer, then compare the writer's dequantized record with the
torch mirror in kv_hadamard.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kv_hadamard import (CAPTURE, FIT, HIDDEN, LATENT, ROWS, WINDOW, Carrier, block_hadamard,  # noqa: E402
                         linear_bf16, nvfp4_two_level, rms_norm)

OUT = Path(__file__).resolve().parents[1] / "results/testC"


def dump() -> None:
    carrier = Carrier()
    P = "model.language_model.layers"
    wkv = carrier.get(f"{P}.19.self_attn.kv_a_proj_with_mqa.weight").cuda()
    gkv = carrier.get(f"{P}.19.self_attn.kv_a_layernorm.weight").cuda()
    gin = carrier.get(f"{P}.19.input_layernorm.weight").cuda().float()
    gpost = carrier.get(f"{P}.18.post_attention_layernorm.weight").cuda().float()
    hid = np.memmap(CAPTURE / "layer-018/hidden.bf16.bin", dtype="<u2", mode="r", shape=(ROWS, HIDDEN))
    rows = []
    for window in FIT["fit_window_indices"][:2]:
        m = torch.from_numpy(np.array(hid[window * WINDOW:(window + 1) * WINDOW], copy=True)).view(torch.bfloat16).cuda()
        a = (gin * (m.float() / gpost)).to(torch.bfloat16)
        rows.append(rms_norm(linear_bf16(a, wkv), gkv))
    c = torch.cat(rows)
    rotated = block_hadamard(c.float(), block_size=LATENT).to(torch.bfloat16)
    torch.save({"kv_c": torch.cat([c, rotated]).cpu()}, OUT / "mirror_inputs.pt")
    print("dumped", tuple(c.shape))


def compare() -> None:
    kv_c = torch.load(OUT / "mirror_inputs.pt")["kv_c"].float()
    kernel = torch.load(OUT / "mirror_kernel_deq.pt")["deq"].float()
    mirror = nvfp4_two_level(kv_c.cuda()).cpu()
    same = (mirror == kernel)
    diff = (mirror - kernel).abs()
    err_k = (kernel - kv_c).double().square().sum()
    err_m = (mirror - kv_c).double().square().sum()
    print(f"elements equal: {same.float().mean().item():.6f}  rows fully equal: {same.all(dim=-1).float().mean().item():.6f}")
    print(f"max |mirror - kernel|: {diff.max().item():.3e}  kernel err energy {err_k.item():.6e}  mirror {err_m.item():.6e}  ratio {(err_m / err_k).item():.6f}")


if __name__ == "__main__":
    dump() if sys.argv[1] == "dump" else compare()
