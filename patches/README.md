# Kernel patches (b12x)

Five unified diffs for the b12x source tree shipped in the serving image
`verdictai/trellismx@sha256:ca6b80188dce154b91f49108b7d87792d2ba6328935afc71b44d1c0e6f6a1adf`.
- The three main patches are relative to that tree as extracted from the image. The 16 existing
  files they modify are byte-identical to the same paths at commit
  `d564f6ca54c092497ec5ae7e07a272f55eec7dbe` of the TrellisMX b12x integration branch.
  `p8_down_remainder.py` is a new file.
- Two small patches stack on the RP2 tree: the host-side guards, then EPI-PAR. Together with the
  RP2 patch they give the kernel tree of the RP2 release image.

| Patch | Applies to | Tree it produces | Used by | Files |
|---|---|---|---|---|
| `b12x-zero-remainder.diff` | the image tree | `b12x-scratch` | Test A (zero-remainder timing arm) and the exploratory FC1/FC2 split | 5 modified |
| `b12x-dx2-rowpack.diff` | the image tree | `b12x-dx2` | Amendment 2 (plane-based D-x2: K2, A2, exploratory split) and Amendments 3-5 (row-packed D-x2-RP: K3, A3, and the `rp` and `rp-fp8` production-window arms) | 15 modified, 1 new (`p8_down_remainder.py`) |
| `b12x-dx2-rowpack-rp2.diff` | the image tree | `b12x-dx2rp2` | Amendment 6 (both-hop row-pack D-x2-RP2: smoke, K4, A4) and the `rp2-fp8` arm of the Amendment 7 run | 16 modified, 1 new (`p8_down_remainder.py`) |
| `b12x-dx2-guards.diff` | `b12x-dx2rp2` | `b12x-dx2rp2-guarded` | the post-audit guard smoke test and the release tree; no KL or timing measurement used it | 1 modified (`p8_native_kernel.py`, host side) |
| `b12x-epipar-m1.diff` | `b12x-dx2rp2-guarded` | the release tree | the RP2 release image and its pre-flight (EPI-PAR in the M1 FC1 kernel) | 2 modified (`route_hoist_k5.py`, `p8_native_kernel.py`) |

The RP2 tree carries the D-x2 and D-x2-RP code plus the RP2 additions. It differs from
`b12x-dx2` in five files: `p8_down_remainder.py`, `route_hoist_direct32.py`,
`route_hoist_dynamic.py`, `route_hoist_k5.py` and `p8_native_kernel.py`.

The three main patches are alternatives: each applies to a clean copy of the extracted b12x tree.
The guards and EPI-PAR patches are a stack on top of the RP2 patch. Apply from the root of the
tree, for example for the release tree:

```
patch -p1 < b12x-dx2-rowpack-rp2.diff
patch -p1 < b12x-dx2-guards.diff
patch -p1 < b12x-epipar-m1.diff
```

The three main diffs were produced with
`diff -ruN -x __pycache__ -x .git -x build -x '*.egg-info' a/ b/`. The guards and EPI-PAR diffs
are plain `diff -r -u -x __pycache__ -x .git` between the two trees; their headers name those
trees.

## Regenerating the trees

The builder scripts regenerate these trees from the image's b12x source. Every text
replacement is counted, and a build aborts if the source does not match.

| Builder | Tree | Output location |
|---|---|---|
| `../scripts/build_zero_remainder_b12x.py` | `b12x-scratch` | set `SRC` and `DST` in the script |
| `../scripts/build_dx2_b12x.py` | `b12x-dx2` | set `SRC` and `DST` in the script |
| `../scripts/build_dx2_rp2_b12x.py` | `b12x-dx2rp2-guarded` by default; `b12x-dx2rp2` with `DX2_GUARDS=0` | set `SRC`; the output directory comes from `DX2_DST` |
| `../scripts/build_rp2_epipar_b12x.py` | the release tree with `EPIPAR_FILES=route_hoist_k5.py`; see below for the check trees | set `SRC` in the RP2 builder; the output directory comes from `EPIPAR_DST` |

`build_dx2_rp2_b12x.py` is the later version of the D-x2 builder, extended with RP2 as
preregistered in Amendment 6. It is kept under its own name so that `build_dx2_b12x.py` still
reproduces the measured `b12x-dx2` tree. Always set `DX2_DST` explicitly: the builder deletes
the destination before writing it, and without the variable it falls back to the `b12x-dx2`
path it was written with. `build_rp2_epipar_b12x.py` runs the RP2 builder into `EPIPAR_DST`
and then adds EPI-PAR.

We checked two routes for each tree:
- **Builders.** Running the builder on the image tree reproduces the measured tree file for
  file, over all 815 files, excluding `__pycache__`, `.git`, `build` and `*.egg-info`.
  - RP2 builder, `DX2_GUARDS=0`: byte-identical to the measured `b12x-dx2rp2` and to
    `b12x-dx2-rowpack-rp2.diff` applied to the image tree.
  - RP2 builder, default (guards on): byte-identical to `b12x-dx2rp2-guarded`, which is
    `b12x-dx2rp2` plus `b12x-dx2-guards.diff`. The two differ only in `p8_native_kernel.py`.
  - EPI-PAR builder, `EPIPAR_FILES=route_hoist_k5.py`: byte-identical to the release image's
    build-context tree (including `build` and `*.egg-info`) and to the RP2, guards and EPI-PAR
    patches applied in turn.
  - EPI-PAR builder, `DX2_GUARDS=0`: byte-identical to the two trees of the EPI-PAR check,
    `b12x-rp2par` (no `EPIPAR_FILES`: M1 and M2-16 kernels) and `b12x-rp2par-k5`
    (`EPIPAR_FILES=route_hoist_k5.py`: M1 only). Those trees predate the guards.
  - `build_dx2_b12x.py`: byte-identical to `b12x-dx2-rowpack.diff` applied to the image tree.
- **Patches.** Applying each patch as described above reproduces the same tree, with no
  rejects.

Neither measured tree changed during the runs that used it:
- **`b12x-dx2`** was built once, before either production window. Both the `rp` (NVFP4 KV)
  and `rp-fp8` (FP8 KV) arms mounted it unchanged.
- **`b12x-dx2rp2`** was built once, before the RP2 smoke test and the K4 and A4 runs, and has
  not changed since. The Amendment 7 `rp2-fp8` arm mounted the same tree.

## Gating

All new kernel code sits behind `cutlass.const_expr` flags:
- `p8_zero_remainder` and `p8_zr_unmask` on `...ZR` / `...ZRU`;
- `p8_down_remainder` on `...DX2`;
- `p8_dx2_rowpack` on `...RP` (and `...RP2`);
- `p8_input_rowpack` has two settings:
  - on the `...RP2` FC1 classes;
  - on the MoE backend instance, for the input prologue, when RP2 is requested;
- `p8_epi_par` on the FC1 kernel that EPI-PAR patches (release tree: the M1 kernel only).

With a flag off, the kernel traces exactly as before. The host-side changes do nothing unless
`B12X_P8_ZERO_REMAINDER`, `B12X_P8_DOWN_REMAINDER` (`1`, `rp` or `rp2`) or `B12X_P8_EPI_PAR=1` is
set when the MoE runtime is constructed, and the scratch layouts default to one plane.

Flag-off behaviour was checked on real outputs:
- **D-x2 build.** With the flag off, outputs were bit-identical to the image's own kernels
  (layers 3 and 8, M16 and M3072, all four TP ranks summed; K2 in `../RESULTS.md`).
- **D-x2-RP.** Only the M <= 16 direct paths are row-packed. On the M64 and M3072 paths, RP
  outputs were bit-identical to prod (K3).
- **D-x2-RP2.** Only the M <= 16 direct paths are row-packed. On the M64 and M3072 paths,
  RP2 outputs were bit-identical to prod (K4), and the smoke test found M64 and M512
  bit-identical as well.
- **Zero-remainder arm.** Its outputs were bit-identical to the flag-off baseline in every
  Test A cell (the extra MMA adds exactly +0), and the unmasked positive control changed
  the outputs.

## Host guards (`b12x-dx2-guards.diff`, added after the 09-26 audit)

The remainder code supports only the configuration the measured runs used:
- `fc1_broadcast_a=True` for RP2;
- `fc1_warp_quant=False` for every remainder mode;
- one route per direct tile (`tile_major_tasks=False`, `grouped_m16=False`).

Other settings could race or silently drop the remainder. None of them was used.

- **What the guards do.** They are host-side checks in `p8_native_kernel.py`. They reject:
  - RP2 without `fc1_broadcast_a`;
  - any remainder mode with `fc1_warp_quant`;
  - multi-route direct tiles.
- **On by default.** The guards were first opt-in. Since the audit's third round,
  `../scripts/build_dx2_rp2_b12x.py` adds them unless `DX2_GUARDS=0`. The same tree comes from
  applying `b12x-dx2-guards.diff` on top of the `b12x-dx2rp2` tree with `patch -p1` from the tree
  root. The release image has them.
- **Measured trees stay reproducible.** With `DX2_GUARDS=0`, the builder's output is
  byte-identical to the measured tree. The guards touch no kernel code.
- **Tested.** `../results/rp2_guard_smoke.json` records a guarded build:
  - the measured configuration passes, with smoke numbers identical to the 09-25 run;
  - all four unsupported combinations tested are rejected.

## EPI-PAR (`b12x-epipar-m1.diff`, exploratory, after Amendment 7)

EPI-PAR parallelizes the down-input quantization in the M1 FC1 epilogue (`route_hoist_k5.py`):
one thread per (row, 32-block) pair instead of one thread per row, with the per-block code
otherwise unchanged. It is switched on by `B12X_P8_EPI_PAR=1` at construction, and the flag
joins the compile spec. The diff also adds that switch to `p8_native_kernel.py`.
- **Identity.** The EPI-PAR check found prod + EPI-PAR equal to prod, and rp2 + EPI-PAR equal to
  rp2, bit for bit on every tested M1, M4 and M16 input (`../results/epipar_check_*.json`).
- **Release pre-flight.** The release image's kernels, with and without EPI-PAR, matched the
  measured outputs in 56/56 cells (`../results/final_identity_check.json`).
- **Why M1 only.** A build that also patched the M2-16 kernel (`route_hoist_direct32.py`) made
  layer 8 at M16 about 5% slower, so the release uses the M1-only build. See section 4.10 of the
  top-level README.

## License

b12x is licensed under the Apache License 2.0 (`LICENSE` in the b12x tree;
`pyproject.toml` declares `license = "Apache-2.0"`). These patches modify b12x source files,
and the upstream license, Apache License 2.0, applies to them. A copy is included as
`LICENSE.b12x`. The ShapleyMCG License in the repository root covers this repository's own
scripts and documents, not these patches.
