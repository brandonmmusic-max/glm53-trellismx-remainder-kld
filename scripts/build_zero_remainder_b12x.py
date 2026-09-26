#!/usr/bin/env python3
"""Build the scratch b12x with the zero-remainder timing arm (Test A, PREREG).

Every replacement is counted; the build aborts on any mismatch. With the flag off
(`p8_zero_remainder = False`, the default and the only value production code sees) the
kernels trace exactly as before: every new statement sits behind cutlass.const_expr.

Arm Z (`...ZR` subclasses, selected only when B12X_P8_ZERO_REMAINDER=1 at compile time):
right after each existing mxfp8 MMA, a second mxfp8_mma_m16n8k32_f32_e4m3 consumes the
SAME decoded weight fragment with an activation fragment that is really loaded from
shared memory (same size/address pattern as the true A fragment, offset by a runtime zero
so it cannot be merged with A's load) and then ANDed with a runtime zero the compiler
cannot prove (expert_idx >> 31). Result: identical loads,
registers and MMA issue as a real remainder term, but the accumulator receives exactly +0.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

SRC = Path("<workspace>/trellismx-review-20260922/image-r27/b12x")
DST = Path("<workspace>/p8-remainder-kvrot-20260923/b12x-scratch")
K = "b12x/moe/_shared/kernels"


def rep(text: str, old: str, new: str, count: int, where: str) -> str:
    found = text.count(old)
    if found != count:
        sys.exit(f"{where}: expected {count} occurrence(s), found {found}: {old[:90]!r}")
    return text.replace(old, new)


FC1_AFRAG_END = """                    a_frag[blk, 0] = a0
                    a_frag[blk, 1] = a1
                    a_frag[blk, 2] = a2
                    a_frag[blk, 3] = a3
                gate_b0 = cute.make_rmem_tensor((self.p8_n8_per_warp,), Uint32)"""
FC1_ZR_LOAD = """                    a_frag[blk, 0] = a0
                    a_frag[blk, 1] = a1
                    a_frag[blk, 2] = a2
                    a_frag[blk, 3] = a3
                if cutlass.const_expr(self.p8_zero_remainder):
                    # Timing arm: a real A-sized smem load, zeroed by a runtime mask.
                    zr_zero = Uint32(expert_idx) >> Uint32(31)
                    zr_mask = zr_zero
                    if cutlass.const_expr(self.p8_zr_unmask):
                        zr_mask = zr_mask | Uint32(0xFFFFFFFF)  # positive control: A_lo = A
                    # Runtime-zero address offset: the compiler cannot merge these loads with A's.
                    zr_off = Int32(zr_zero) << Int32(4)
                    zr_frag = cute.make_rmem_tensor((self.mma_m_blocks, 4), Uint32)
                    for blk in cutlass.range_constexpr(self.mma_m_blocks):
                        zr_lo = a_base + Int32(blk * 16 * 128) + (q << Int32(7)) + (u_phys << Int32(4)) + ((c & Int32(1)) << Int32(3))
                        if cutlass.const_expr(self.p8_broadcast_a):
                            zr_lo = a_base + (u_phys << Int32(4)) + ((c & Int32(1)) << Int32(3))
                        zr_lo = zr_lo + zr_off
                        z0, z2 = ld_shared_v2_u32(zr_lo)
                        if cutlass.const_expr(self.p8_broadcast_a):
                            z1, z3 = (z0, z2)
                        else:
                            z1, z3 = ld_shared_v2_u32(zr_lo + Int32(8 * 128))
                        zr_frag[blk, 0] = z0 & zr_mask
                        zr_frag[blk, 1] = z1 & zr_mask
                        zr_frag[blk, 2] = z2 & zr_mask
                        zr_frag[blk, 3] = z3 & zr_mask
                gate_b0 = cute.make_rmem_tensor((self.p8_n8_per_warp,), Uint32)"""

FC1_GATE_MMA = """                            g0, g1, g2, g3 = mxfp8_mma_m16n8k32_f32_e4m3(gate_fragment[0], gate_fragment[1], gate_fragment[2], gate_fragment[3], a_frag[blk, 0], a_frag[blk, 1], a_frag[blk, 2], a_frag[blk, 3], gb0, gb1, asc[blk], gate_sfb, bid_a=kb, bid_b=kb)
"""
FC1_GATE_MMA_ZR = FC1_GATE_MMA + """                            if cutlass.const_expr(self.p8_zero_remainder):
                                g0, g1, g2, g3 = mxfp8_mma_m16n8k32_f32_e4m3(g0, g1, g2, g3, zr_frag[blk, 0], zr_frag[blk, 1], zr_frag[blk, 2], zr_frag[blk, 3], gb0, gb1, asc[blk], gate_sfb, bid_a=kb, bid_b=kb)
"""
FC1_UP_MMA = """                            u0, u1, u2, u3 = mxfp8_mma_m16n8k32_f32_e4m3(up_fragment[0], up_fragment[1], up_fragment[2], up_fragment[3], a_frag[blk, 0], a_frag[blk, 1], a_frag[blk, 2], a_frag[blk, 3], ub0, ub1, asc[blk], up_sfb, bid_a=kb, bid_b=kb)
"""
FC1_UP_MMA_ZR = FC1_UP_MMA + """                            if cutlass.const_expr(self.p8_zero_remainder):
                                u0, u1, u2, u3 = mxfp8_mma_m16n8k32_f32_e4m3(u0, u1, u2, u3, zr_frag[blk, 0], zr_frag[blk, 1], zr_frag[blk, 2], zr_frag[blk, 3], ub0, ub1, asc[blk], up_sfb, bid_a=kb, bid_b=kb)
"""

FC2_AFRAG_END = """                    a0, a2 = ld_shared_v2_u32(a_lo)
                    a1, a3 = ld_shared_v2_u32(a_lo + Int32(8 * self.tile_k))
                    a_frag[blk, 0] = a0
                    a_frag[blk, 1] = a1
                    a_frag[blk, 2] = a2
                    a_frag[blk, 3] = a3
"""
FC2_ZR_LOAD = FC2_AFRAG_END + """                if cutlass.const_expr(self.p8_zero_remainder):
                    # Timing arm: a real A-sized smem load, zeroed by a runtime mask.
                    zr_zero = Uint32(expert_idx) >> Uint32(31)
                    zr_mask = zr_zero
                    if cutlass.const_expr(self.p8_zr_unmask):
                        zr_mask = zr_mask | Uint32(0xFFFFFFFF)  # positive control: A_lo = A
                    # Runtime-zero address offset: the compiler cannot merge these loads with A's.
                    zr_off = Int32(zr_zero) << Int32(4)
                    zr_frag = cute.make_rmem_tensor((self.mma_m_blocks, 4), Uint32)
                    for blk in cutlass.range_constexpr(self.mma_m_blocks):
                        zr_lo = (
                            a_base
                            + Int32(blk * 16 * self.tile_k)
                            + (q << Int32(7))
                            + (u_phys << Int32(4))
                            + ((c & Int32(1)) << Int32(3))
                            + zr_off
                        )
                        z0, z2 = ld_shared_v2_u32(zr_lo)
                        z1, z3 = ld_shared_v2_u32(zr_lo + Int32(8 * self.tile_k))
                        zr_frag[blk, 0] = z0 & zr_mask
                        zr_frag[blk, 1] = z1 & zr_mask
                        zr_frag[blk, 2] = z2 & zr_mask
                        zr_frag[blk, 3] = z3 & zr_mask
"""
FC2_MMA_END = """                                sfb_word,
                                bid_a=kb,
                                bid_b=kb,
                            )
                        fragment[0] = d0"""
FC2_MMA_END_ZR = """                                sfb_word,
                                bid_a=kb,
                                bid_b=kb,
                            )
                            if cutlass.const_expr(self.p8_zero_remainder):
                                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(d0, d1, d2, d3, zr_frag[blk, 0], zr_frag[blk, 1], zr_frag[blk, 2], zr_frag[blk, 3], b0, b1, asc[blk], sfb_word, bid_a=kb, bid_b=kb)
                        fragment[0] = d0"""

FLAG = "    mma_m_blocks = 1\n"
FLAG_NEW = "    mma_m_blocks = 1\n    p8_zero_remainder = False\n    p8_zr_unmask = False\n"

FILES = {
    "route_hoist_direct32.py": ("fc1", "P8DirectM32FC1Kernel"),
    "route_hoist_k5.py": ("fc1", "P8H128FC1Kernel"),
    "direct_m32_fc2.py": ("fc2", "P8DirectM32FC2Kernel"),
    "fc2_k5_funnel.py": ("fc2", "P8SmallMPhase2Kernel"),
}

SEL_FC2 = """            if mma_tiler_mn == (32, 128):
                from b12x.moe._shared.kernels.direct_m32_fc2 import P8DirectM32FC2Kernel as P8SmallMPhase2Kernel
"""
# B12X_P8_ZERO_REMAINDER: 1 both ZR, 2 both ZRU, 3 FC2-only ZR, 4 FC1-only ZR, 5 FC2-only ZRU
SEL_FC2_NEW = SEL_FC2 + """            _zr = os.environ.get("B12X_P8_ZERO_REMAINDER")
            if _zr in ("1", "2", "3", "5"):
                import importlib
                _sfx = "ZRU" if _zr in ("2", "5") else "ZR"
                if mma_tiler_mn == (32, 128):
                    P8SmallMPhase2Kernel = getattr(importlib.import_module("b12x.moe._shared.kernels.direct_m32_fc2"), "P8DirectM32FC2Kernel" + _sfx)
                else:
                    P8SmallMPhase2Kernel = getattr(importlib.import_module("b12x.moe._shared.kernels.fc2_k5_funnel"), "P8SmallMPhase2Kernel" + _sfx)
"""
SEL_FC1 = """            if mma_tiler_mn == (32, 128):
                from b12x.moe._shared.kernels.route_hoist_direct32 import P8DirectM32FC1Kernel as P8H128FC1Kernel
"""
SEL_FC1_NEW = SEL_FC1 + """            _zr = os.environ.get("B12X_P8_ZERO_REMAINDER")
            if _zr in ("1", "2", "4"):
                import importlib
                _sfx = "ZRU" if _zr == "2" else "ZR"
                if mma_tiler_mn == (32, 128):
                    P8H128FC1Kernel = getattr(importlib.import_module("b12x.moe._shared.kernels.route_hoist_direct32"), "P8DirectM32FC1Kernel" + _sfx)
                else:
                    P8H128FC1Kernel = getattr(importlib.import_module("b12x.moe._shared.kernels.route_hoist_k5"), "P8H128FC1Kernel" + _sfx)
"""

def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "build", "*.egg-info"))
    for name, (kind, cls) in FILES.items():
        path = DST / K / name
        s = path.read_text()
        s = rep(s, FLAG, FLAG_NEW, 1, name)
        if kind == "fc1":
            s = rep(s, FC1_AFRAG_END, FC1_ZR_LOAD, 1, name)
            s = rep(s, FC1_GATE_MMA, FC1_GATE_MMA_ZR, 1, name)
            s = rep(s, FC1_UP_MMA, FC1_UP_MMA_ZR, 1, name)
        else:
            s = rep(s, FC2_AFRAG_END, FC2_ZR_LOAD, 1, name)
            s = rep(s, FC2_MMA_END, FC2_MMA_END_ZR, 1, name)
        if f"\nclass {cls}(" not in s:
            sys.exit(f"{name}: class {cls} not found")
        s += f"\n\nclass {cls}ZR({cls}):\n    \"\"\"Zero-remainder timing arm (PREREG Test A).\"\"\"\n\n    p8_zero_remainder = True\n"
        s += f"\n\nclass {cls}ZRU({cls}ZR):\n    \"\"\"Positive control: the extra MMA consumes the real A fragment.\"\"\"\n\n    p8_zr_unmask = True\n"
        path.write_text(s)
    path = DST / K / "route_hoist_dynamic.py"
    s = path.read_text()
    s = rep(s, SEL_FC2, SEL_FC2_NEW, 1, "route_hoist_dynamic.py")
    s = rep(s, SEL_FC1, SEL_FC1_NEW, 1, "route_hoist_dynamic.py")
    path.write_text(s)
    print(f"built {DST}")


if __name__ == "__main__":
    main()
