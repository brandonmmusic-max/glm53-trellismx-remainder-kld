#!/usr/bin/env python3
"""Build a scratch b12x with the real down-hop remainder (D-x2, PREREG amendment 2).

Every replacement is counted and the build aborts on any mismatch. All new kernel code sits
behind `cutlass.const_expr(getattr(self, "p8_down_remainder", False))`, so classes without the
flag trace exactly as before. D-x2 is switched on per P8NativeTPMoE instance by
B12X_P8_DOWN_REMAINDER=1 at construction. The flag joins the compile spec, and the phase
kernels are upgraded to the statically defined `...DX2` subclasses.

D-x2 numerics:
- The FC1 epilogue quantizes each 32-block of the down input as today (`hi`, permuted K
  order, UE8M0 = pow2_ceil(amax/448)).
- It then writes `lo = q(v - dq(hi))` with its own UE8M0 scale into a second plane of the
  intermediate buffer, laid out exactly like the first and starting right after it.
- FC2 stages both planes and issues a second mxfp8 MMA per decoded weight fragment (lo A,
  lo scale, same B/SFB) into the same FP32 accumulator.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

SRC = Path("<workspace>/trellismx-review-20260922/image-r27/b12x")
DST = Path("<workspace>/p8-remainder-kvrot-20260923/b12x-dx2")
K = "b12x/moe/_shared/kernels"
T = "b12x/moe/_shared/trellismx"
FLAG = 'cutlass.const_expr(getattr(self, "p8_down_remainder", False))'


def rep(text: str, old: str, new: str, count: int, where: str) -> str:
    found = text.count(old)
    if found != count:
        sys.exit(f"{where}: expected {count} occurrence(s), found {found}: {old[:100]!r}")
    return text.replace(old, new)


# ---------------------------------------------------------------- FC1 epilogue (lo plane writer)
DX2_FC1 = f"""                if {FLAG}:
                    # D-x2: second E4M3/UE8M0-K32 plane, lo = v - dq(hi), after the whole hi region.
                    dx2_vals = values
                    if cutlass.const_expr(self.w4a8_trellis):
                        dx2_vals = _w4a8_trellis_permute_k32(values)
                    dx2_scale, _ = pow2_ceil_ue8m0(block_max * cutlass.Float32(1.0 / 448.0))
                    dx2_lo = cute.make_rmem_tensor((32,), cutlass.Float32)
                    dx2_max = cutlass.Float32(0.0)
                    for dx2_w in cutlass.range_constexpr(8):
                        dx2_h0, dx2_h1, dx2_h2, dx2_h3 = cvt_e4m3x4_to_f32x4(payload[dx2_w])
                        dx2_lo[dx2_w * 4 + 0] = dx2_vals[dx2_w * 4 + 0] - dx2_h0 * dx2_scale
                        dx2_lo[dx2_w * 4 + 1] = dx2_vals[dx2_w * 4 + 1] - dx2_h1 * dx2_scale
                        dx2_lo[dx2_w * 4 + 2] = dx2_vals[dx2_w * 4 + 2] - dx2_h2 * dx2_scale
                        dx2_lo[dx2_w * 4 + 3] = dx2_vals[dx2_w * 4 + 3] - dx2_h3 * dx2_scale
                    for dx2_e in cutlass.range_constexpr(32):
                        dx2_abs = fabs_f32(dx2_lo[dx2_e])
                        if dx2_abs > dx2_max:
                            dx2_max = dx2_abs
                    dx2_payload, dx2_byte = quantize_block_fp8_mx(dx2_lo, dx2_max)
                    dx2_off = rows_capacity * (words_per_row + intermediate_tiles)
                    for dx2_w in cutlass.range_constexpr(8):
                        intermediate_u32[dx2_off + dst_word + Int32(dx2_w)] = dx2_payload[dx2_w]
                    scale_bytes[(dx2_off + sf_base + output_tile * rows_capacity + physical_row_base + tid) * Int32(4) + block] = cutlass.Uint8(dx2_byte & Uint32(255))
"""
RP_FLAG = 'cutlass.const_expr(getattr(self, "p8_dx2_rowpack", False))'
RP_FC1 = f"""                if {RP_FLAG}:
                    # D-x2-RP: lo = v - dq(hi) into row +8 of this route's own tile (direct tiles hold one route).
                    rp_vals = values
                    if cutlass.const_expr(self.w4a8_trellis):
                        rp_vals = _w4a8_trellis_permute_k32(values)
                    rp_scale, _ = pow2_ceil_ue8m0(block_max * cutlass.Float32(1.0 / 448.0))
                    rp_lo = cute.make_rmem_tensor((32,), cutlass.Float32)
                    rp_max = cutlass.Float32(0.0)
                    for rp_w in cutlass.range_constexpr(8):
                        rp_h0, rp_h1, rp_h2, rp_h3 = cvt_e4m3x4_to_f32x4(payload[rp_w])
                        rp_lo[rp_w * 4 + 0] = rp_vals[rp_w * 4 + 0] - rp_h0 * rp_scale
                        rp_lo[rp_w * 4 + 1] = rp_vals[rp_w * 4 + 1] - rp_h1 * rp_scale
                        rp_lo[rp_w * 4 + 2] = rp_vals[rp_w * 4 + 2] - rp_h2 * rp_scale
                        rp_lo[rp_w * 4 + 3] = rp_vals[rp_w * 4 + 3] - rp_h3 * rp_scale
                    for rp_e in cutlass.range_constexpr(32):
                        rp_abs = fabs_f32(rp_lo[rp_e])
                        if rp_abs > rp_max:
                            rp_max = rp_abs
                    rp_payload, rp_byte = quantize_block_fp8_mx(rp_lo, rp_max)
                    rp_dst = dst_word + Int32(8) * words_per_row
                    for rp_w in cutlass.range_constexpr(8):
                        intermediate_u32[rp_dst + Int32(rp_w)] = rp_payload[rp_w]
                    scale_bytes[(sf_base + output_tile * rows_capacity + physical_row_base + tid + Int32(8)) * Int32(4) + block] = cutlass.Uint8(rp_byte & Uint32(255))
"""
FC1_STORE = """                scale_bytes[(sf_base + output_tile * rows_capacity + physical_row_base + tid) * Int32(4) + block] = cutlass.Uint8(scale_byte & Uint32(255))
"""
FC1_STORE_ALIAS = """                scale_bytes[
                    (sf_base + output_tile * rows_capacity + physical_row_base + tid)
                    * Int32(4) + block
                ] = cutlass.Uint8(scale_byte & Uint32(0xFF))
"""
FC1_IMPORT = "import cutlass.cute as cute\n"
FC1_IMPORT_NEW = "import cutlass.cute as cute\nfrom b12x._lib.intrinsics import cvt_e4m3x4_to_f32x4\n"

# ---------------------------------------------------------------- FC2 body (lo A + second MMA)
FC2_ASC = """                asc[blk] = ld_shared_u32(sfa_base + (sf_row << Int32(2)))
"""
FC2_ASC_NEW = FC2_ASC + f"""            if {FLAG}:
                dx2_a_base = smem_base + Int32(self.dx2_a_offset) + stage * Int32(self.a_stage_bytes)
                dx2_sfa_base = dx2_a_base + Int32(self.a_payload_bytes)
                dx2_asc = cute.make_rmem_tensor((self.mma_m_blocks,), Uint32)
                for blk in cutlass.range_constexpr(self.mma_m_blocks):
                    sf_row = Int32(blk * 16) + q + ((lane & Int32(1)) << Int32(3))
                    dx2_asc[blk] = ld_shared_u32(dx2_sfa_base + (sf_row << Int32(2)))
"""
FC2_AFRAG_END = """                    a0, a2 = ld_shared_v2_u32(a_lo)
                    a1, a3 = ld_shared_v2_u32(a_lo + Int32(8 * self.tile_k))
                    a_frag[blk, 0] = a0
                    a_frag[blk, 1] = a1
                    a_frag[blk, 2] = a2
                    a_frag[blk, 3] = a3
"""
FC2_AFRAG_NEW = FC2_AFRAG_END + f"""                if {FLAG}:
                    dx2_frag = cute.make_rmem_tensor((self.mma_m_blocks, 4), Uint32)
                    for blk in cutlass.range_constexpr(self.mma_m_blocks):
                        dx2_a_lo = (
                            dx2_a_base
                            + Int32(blk * 16 * self.tile_k)
                            + (q << Int32(7))
                            + (u_phys << Int32(4))
                            + ((c & Int32(1)) << Int32(3))
                        )
                        dx2_z0, dx2_z2 = ld_shared_v2_u32(dx2_a_lo)
                        dx2_z1, dx2_z3 = ld_shared_v2_u32(dx2_a_lo + Int32(8 * self.tile_k))
                        dx2_frag[blk, 0] = dx2_z0
                        dx2_frag[blk, 1] = dx2_z1
                        dx2_frag[blk, 2] = dx2_z2
                        dx2_frag[blk, 3] = dx2_z3
"""
FC2_MMA_END = """                                sfb_word,
                                bid_a=kb,
                                bid_b=kb,
                            )
                        fragment[0] = d0"""
FC2_MMA_NEW = f"""                                sfb_word,
                                bid_a=kb,
                                bid_b=kb,
                            )
                            if {FLAG}:
                                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(d0, d1, d2, d3, dx2_frag[blk, 0], dx2_frag[blk, 1], dx2_frag[blk, 2], dx2_frag[blk, 3], b0, b1, dx2_asc[blk], sfb_word, bid_a=kb, bid_b=kb)
                        fragment[0] = d0"""

FC2_ROWS = """                    fragment = facc[blk][nt]
                    row_lo = Int32(blk * 16) + q
                    row_hi = row_lo + Int32(8)
"""
FC2_ROWS_NEW = FC2_ROWS + f"""                    if {RP_FLAG}:
                        # D-x2-RP: row q+8 carries lo*B for the route in row q.
                        fragment[0] = fragment[0] + fragment[2]
                        fragment[1] = fragment[1] + fragment[3]
"""

# ---------------------------------------------------------------- FC2 staging (inherited by all FC2s)
STAGE_SFA = """            cp_async_u32_shared_global(
                sfa_base + (tid << Int32(2)),
                get_ptr_as_int64(intermediate_u32, sf_src),
            )
"""
STAGE_SFA_NEW = STAGE_SFA + f"""
        if {FLAG}:
            # D-x2: stage the lo plane (hi layout, offset by the whole hi region).
            dx2_off = rows_capacity * (words_per_row + intermediate_tiles)
            dx2_a_base = smem_base + Int32(self.dx2_a_offset) + stage * Int32(self.a_stage_bytes)
            dx2_sfa_base = dx2_a_base + Int32(self.a_payload_bytes)
            for i in cutlass.range_constexpr(
                (self.tile_m * 8 + self.threads_per_cta - 1) // self.threads_per_cta
            ):
                idx = tid + Int32(i * self.threads_per_cta)
                if idx < Int32(self.tile_m * 8):
                    row = idx >> Int32(3)
                    vec = idx & Int32(7)
                    physical_vec = vec ^ (row & Int32(7))
                    src_word = (
                        dx2_off
                        + (physical_row_base + row) * words_per_row
                        + intermediate_slice * Int32(32)
                        + (vec << Int32(2))
                    )
                    cp_async4_shared_global(
                        dx2_a_base + row * Int32(self.tile_k) + (physical_vec << Int32(4)),
                        get_ptr_as_int64(intermediate_u32, src_word),
                    )
            if tid < Int32(self.tile_m):
                dx2_sf_src = (
                    dx2_off
                    + rows_capacity * words_per_row
                    + intermediate_slice * rows_capacity
                    + physical_row_base
                    + tid
                )
                cp_async_u32_shared_global(
                    dx2_sfa_base + (tid << Int32(2)),
                    get_ptr_as_int64(intermediate_u32, dx2_sf_src),
                )
"""

ENABLE_MODULE = '''"""D-x2: two-term (hi + lo) E4M3/UE8M0-K32 carrier at the P8 down (FC2) input only."""
from __future__ import annotations

import importlib


def enable_down_remainder(backend, phases: str = "both") -> None:
    """Upgrade a constructed backend's phase kernels to their statically defined DX2 classes.

    `phases` is a diagnostic split: "fc1" only writes the lo plane (FC2 ignores it), "fc2" only
    stages the (zero) lo plane and issues the extra MMA. Both must be bit-identical to prod.
    """
    if phases not in ("both", "fc1", "fc2"):
        raise ValueError(f"unknown D-x2 phase split: {phases!r}")
    attrs = {"both": ("materialized_phase1_kernel", "materialized_phase2_kernel"),
             "fc1": ("materialized_phase1_kernel",), "fc2": ("materialized_phase2_kernel",)}[phases]
    for attr in attrs:
        obj = getattr(backend, attr)
        cls = type(obj)
        dx2 = getattr(importlib.import_module(cls.__module__), cls.__name__ + "DX2", None)
        if dx2 is None or not issubclass(dx2, cls):
            raise RuntimeError(f"no D-x2 variant for {cls.__module__}.{cls.__name__}")
        obj.__class__ = dx2
    if phases == "fc1":
        return
    fc2 = backend.materialized_phase2_kernel
    # FC2 double-buffers the lo A tile + lo scales after everything it already owns.
    fc2.dx2_a_offset = (int(fc2.shared_bytes) + 1023) // 1024 * 1024
    fc2.shared_bytes = fc2.dx2_a_offset + 2 * int(fc2.a_stage_bytes)
    fc2.shared_words = (fc2.shared_bytes + 3) // 4
    fc2.trellis_lut_offset = fc2.shared_bytes


def enable_rowpack(backend) -> None:
    """D-x2-RP: upgrade direct (M<=16) phase kernels to their static RP classes. No smem change."""
    for attr in ("materialized_phase1_kernel", "materialized_phase2_kernel"):
        obj = getattr(backend, attr)
        cls = type(obj)
        rp = getattr(importlib.import_module(cls.__module__), cls.__name__ + "RP", None)
        if rp is None or not issubclass(rp, cls):
            raise RuntimeError(f"no D-x2-RP variant for {cls.__module__}.{cls.__name__}")
        obj.__class__ = rp
'''

DX2_CLASSES = {
    "route_hoist_k5.py": "P8H128FC1Kernel",
    "route_hoist_direct32.py": "P8DirectM32FC1Kernel",
    "route_hoist_prefill.py": "P8CoupledPrefillFC1Kernel",
    "policy_m32_fc1.py": "P8CoupledPrefillFC1Kernel",
    "fc2_k5_funnel.py": "P8SmallMPhase2Kernel",
    "direct_m32_fc2.py": "P8DirectM32FC2Kernel",
    "grouped_fc2_grid376.py": "P8CoupledPrefillFC2Kernel",
    "policy_m32_fc2.py": "P8CoupledPrefillFC2Kernel",
}


RP_CLASSES = {
    "route_hoist_k5.py": "P8H128FC1Kernel",
    "route_hoist_direct32.py": "P8DirectM32FC1Kernel",
    "fc2_k5_funnel.py": "P8SmallMPhase2Kernel",
    "direct_m32_fc2.py": "P8DirectM32FC2Kernel",
}


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
    k = DST / K
    (k / "p8_down_remainder.py").write_text(ENABLE_MODULE)

    for name in ("route_hoist_k5.py", "route_hoist_direct32.py", "route_hoist_fc1.py", "row_alias376_fc1.py"):
        p = k / name
        s = rep(p.read_text(), FC1_IMPORT, FC1_IMPORT_NEW, 1, name)
        store = FC1_STORE_ALIAS if name == "row_alias376_fc1.py" else FC1_STORE
        s = rep(s, store, store + DX2_FC1 + ("" if name == "row_alias376_fc1.py" else RP_FC1), 1, name)
        p.write_text(s)

    for name in ("fc2_k5_funnel.py", "direct_m32_fc2.py", "p8_small_m.py"):
        p = k / name
        s = p.read_text()
        s = rep(s, FC2_ASC, FC2_ASC_NEW, 1, name)
        s = rep(s, FC2_AFRAG_END, FC2_AFRAG_NEW, 1, name)
        s = rep(s, FC2_MMA_END, FC2_MMA_NEW, 1, name)
        s = rep(s, FC2_ROWS, FC2_ROWS_NEW, 1, name)
        p.write_text(s)

    p = k / "w4a8_phase2.py"
    p.write_text(rep(p.read_text(), STAGE_SFA, STAGE_SFA_NEW, 1, "w4a8_phase2.py"))

    for name, cls in DX2_CLASSES.items():
        p = k / name
        s = p.read_text()
        if f"\nclass {cls}(" not in s:
            sys.exit(f"{name}: class {cls} not found")
        s += (f"\n\nclass {cls}DX2({cls}):\n    \"\"\"D-x2 down-hop remainder (PREREG amendment 2).\"\"\"\n\n"
              f"    p8_down_remainder = True\n")
        p.write_text(s)

    for name, cls in RP_CLASSES.items():
        p = k / name
        s = p.read_text()
        s += (f"\n\nclass {cls}RP({cls}):\n    \"\"\"D-x2-RP row-packed down-hop remainder (PREREG amendment 3).\"\"\"\n\n"
              f"    p8_dx2_rowpack = True\n")
        p.write_text(s)

    # Scratch layouts: the intermediate region holds `planes` copies (hi, lo).
    p = DST / T / "policy_smallm_schedule.py"
    s = p.read_text()
    s = rep(s, "tile_m: int | None = None, direct: bool | None = None) -> P8ScratchLayout:",
            "tile_m: int | None = None, direct: bool | None = None, planes: int = 1) -> P8ScratchLayout:",
            1, "policy_smallm_schedule.py")
    s = rep(s, "(rows * (geometry.intermediate + geometry.intermediate // 32) // 4,)),",
            "(planes * rows * (geometry.intermediate + geometry.intermediate // 32) // 4,)),",
            1, "policy_smallm_schedule.py")
    p.write_text(s)
    p = DST / T / "direct_policy_scratch.py"
    s = p.read_text()
    s = rep(s, "def direct_scratch_layout(tokens, intermediate=512, *, tile_m=16):",
            "def direct_scratch_layout(tokens, intermediate=512, *, tile_m=16, planes=1):", 1, "direct_policy_scratch.py")
    s = rep(s, "shared=True, grouped=False, tile_m=tile_m, direct=True)",
            "shared=True, grouped=False, tile_m=tile_m, direct=True, planes=planes)", 1, "direct_policy_scratch.py")
    p.write_text(s)

    # Runtime: flag captured at construction, joins the compile spec, sizes the buffers.
    p = DST / T / "p8_native_kernel.py"
    s = p.read_text()
    s = rep(s, "        self.device = torch.device(device)\n",
            "        self.device = torch.device(device)\n"
            "        # D-x2 (PREREG amendment 2): captured once; joins the compile spec below.\n"
            "        self.p8_down_remainder = os.environ.get(\"B12X_P8_DOWN_REMAINDER\") == \"1\"\n"
            "        self._dx2_planes = 2 if self.p8_down_remainder else 1\n"
            "        self._dx2_phases = os.environ.get(\"B12X_P8_DOWN_REMAINDER_PHASES\", \"both\")\n"
            "        self.p8_dx2_rowpack = os.environ.get(\"B12X_P8_DOWN_REMAINDER\") == \"rp\"\n"
            "        if self.p8_dx2_rowpack:\n"
            "            print(f\"P8_DX2_ROWPACK_ACTIVE layer={layer} rank={tp_rank}\", flush=True)\n",
            1, "p8_native_kernel.py")
    s = rep(s, "        if self.diagnostic_raw_fc1:\n            if not (small_m and self.full_coupled):",
            "        if self.p8_down_remainder:\n"
            "            if not self.full_coupled:\n"
            "                raise RuntimeError(\"D-x2 is implemented for the full-coupled P8 paths only\")\n"
            "            from b12x.moe._shared.kernels.p8_down_remainder import enable_down_remainder\n"
            "            enable_down_remainder(kernel, self._dx2_phases)\n"
            "        if self.p8_dx2_rowpack and small_m:\n"
            "            if not self.full_coupled:\n"
            "                raise RuntimeError(\"D-x2-RP is implemented for the full-coupled P8 paths only\")\n"
            "            from b12x.moe._shared.kernels.p8_down_remainder import enable_rowpack\n"
            "            enable_rowpack(kernel)\n"
            "        if self.diagnostic_raw_fc1:\n            if not (small_m and self.full_coupled):",
            1, "p8_native_kernel.py")
    s = rep(s, "                (\"deterministic_output\", int(self.deterministic_output)),\n            ),\n            dsl_compile_options=OptLevel(2),",
            "                (\"deterministic_output\", int(self.deterministic_output)),\n"
            "                (\"down_remainder\", int(self.p8_down_remainder)),\n"
            "                (\"down_remainder_phases\", self._dx2_phases if self.p8_down_remainder else \"none\"),\n"
            "                (\"dx2_rowpack\", int(self.p8_dx2_rowpack and small_m)),\n"
            "            ),\n            dsl_compile_options=OptLevel(2),",
            1, "p8_native_kernel.py")
    s = rep(s, "direct_scratch_layout(m, self.intermediate, tile_m=tile_m) if small_m",
            "direct_scratch_layout(m, self.intermediate, tile_m=tile_m, planes=self._dx2_planes) if small_m",
            1, "p8_native_kernel.py")
    s = rep(s, "else p8_small_m_scratch_layout(intermediate=self.intermediate, tokens=m,",
            "else p8_small_m_scratch_layout(intermediate=self.intermediate, tokens=m, planes=self._dx2_planes,",
            1, "p8_native_kernel.py")
    s = rep(s, "layout = (p8_small_m_scratch_layout(intermediate=self.intermediate, tokens=m, shared=True,",
            "layout = (p8_small_m_scratch_layout(intermediate=self.intermediate, tokens=m, shared=True, planes=self._dx2_planes,",
            1, "p8_native_kernel.py")
    s = rep(s, "intermediate_count = rows_padded * (self.intermediate + self.intermediate // 32) // 4\n",
            "intermediate_count = self._dx2_planes * rows_padded * (self.intermediate + self.intermediate // 32) // 4\n",
            1, "p8_native_kernel.py")
    p.write_text(s)
    print(f"built {DST}")


if __name__ == "__main__":
    main()
