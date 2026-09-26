# Kernel patches (b12x)

Three unified diffs against the b12x source tree shipped in the serving image
`verdictai/trellismx@sha256:ca6b80188dce154b91f49108b7d87792d2ba6328935afc71b44d1c0e6f6a1adf`,
plus one small opt-in guards diff (see "Opt-in guards" below). The three main patches are
relative to that tree as extracted from the image.
- The 16 existing files they modify are byte-identical to the same paths at commit
  `d564f6ca54c092497ec5ae7e07a272f55eec7dbe` of the TrellisMX b12x integration branch.
- `p8_down_remainder.py` is a new file.

| Patch | Tree it produces | Used by | Files |
|---|---|---|---|
| `b12x-zero-remainder.diff` | `b12x-scratch` | Test A (zero-remainder timing arm) and the exploratory FC1/FC2 split | 5 modified |
| `b12x-dx2-rowpack.diff` | `b12x-dx2` | Amendment 2 (plane-based D-x2: K2, A2, exploratory split) and Amendments 3-5 (row-packed D-x2-RP: K3, A3, and the `rp` and `rp-fp8` production-window arms) | 15 modified, 1 new (`p8_down_remainder.py`) |
| `b12x-dx2-rowpack-rp2.diff` | `b12x-dx2rp2` | Amendment 6 (both-hop row-pack D-x2-RP2: smoke, K4, A4) and the `rp2-fp8` arm of the Amendment 7 run | 16 modified, 1 new (`p8_down_remainder.py`) |
| `b12x-dx2-guards.diff` | `b12x-dx2rp2` plus opt-in guards (applied on top of `b12x-dx2rp2`) | the post-audit guard smoke test only; no measurement used it | 1 modified (`p8_native_kernel.py`, host side) |

The RP2 tree carries the D-x2 and D-x2-RP code plus the RP2 additions. It differs from
`b12x-dx2` in five files: `p8_down_remainder.py`, `route_hoist_direct32.py`,
`route_hoist_dynamic.py`, `route_hoist_k5.py` and `p8_native_kernel.py`.

Each patch applies to a clean copy of the extracted b12x tree. They are alternatives, not a
stack. Apply from the root of that tree:

```
patch -p1 < b12x-dx2-rowpack-rp2.diff
```

The diffs were produced with
`diff -ruN -x __pycache__ -x .git -x build -x '*.egg-info' a/ b/`.

## Regenerating the trees

The builder scripts regenerate these trees from the image's b12x source. Every text
replacement is counted, and a build aborts if the source does not match.

| Builder | Tree | Output location |
|---|---|---|
| `../scripts/build_zero_remainder_b12x.py` | `b12x-scratch` | set `SRC` and `DST` in the script |
| `../scripts/build_dx2_b12x.py` | `b12x-dx2` | set `SRC` and `DST` in the script |
| `../scripts/build_dx2_rp2_b12x.py` | `b12x-dx2rp2` | set `SRC`; the output directory comes from `DX2_DST` |

`build_dx2_rp2_b12x.py` is the later version of the D-x2 builder, extended with RP2 as
preregistered in Amendment 6. It is kept under its own name so that `build_dx2_b12x.py` still
reproduces the measured `b12x-dx2` tree. Always set `DX2_DST` explicitly: the builder deletes
the destination before writing it, and without the variable it falls back to the `b12x-dx2`
path it was written with.

We checked two routes for each tree:
- **Builders.** Running the builder on the image tree reproduces the measured tree file for
  file.
  - For RP2 the check was `DX2_DST=<scratch dir>`, compared with the measured `b12x-dx2rp2`
    over all 815 files, excluding `__pycache__`, `.git`, `build` and `*.egg-info`.
  - After the audit it was repeated with the current builder, which contains the opt-in guards.
    Without `DX2_GUARDS`, its output is still byte-identical to the measured `b12x-dx2rp2` and
    to `b12x-dx2-rowpack-rp2.diff` applied to the image tree.
- **Patches.** Applying each patch to a clean copy of the image tree reproduces the same tree,
  with no rejects.

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
  - on the MoE backend instance, for the input prologue, when RP2 is requested.

With a flag off, the kernel traces exactly as before. The host-side changes do nothing unless
`B12X_P8_ZERO_REMAINDER` or `B12X_P8_DOWN_REMAINDER` (`1`, `rp` or `rp2`) is set when the MoE
runtime is constructed, and the scratch layouts default to one plane.

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

## Opt-in guards (`b12x-dx2-guards.diff`, added after the 09-26 audit)

The remainder code supports only the configuration the measured runs used:
- `fc1_broadcast_a=True` for RP2;
- `fc1_warp_quant=False` for every remainder mode;
- one route per direct tile (`tile_major_tasks=False`, `grouped_m16=False`).

Other settings could race or silently drop the remainder. None of them was used.

- **What the guards do.** They are host-side checks in `p8_native_kernel.py`. They reject:
  - RP2 without `fc1_broadcast_a`;
  - any remainder mode with `fc1_warp_quant`;
  - multi-route direct tiles.
- **How to enable them.** Build with `DX2_GUARDS=1 DX2_DST=<scratch dir>
  ../scripts/build_dx2_rp2_b12x.py`, or apply `b12x-dx2-guards.diff` on top of the `b12x-dx2rp2`
  tree. It is a plain `diff -ru` between the two trees, so use `patch -p1` from the tree root.
  Both routes give the same tree.
- **Default builds are unchanged.** Without `DX2_GUARDS`, the builder's output is byte-identical
  to the measured trees. The guards touch no kernel code.
- **Tested.** `../results/rp2_guard_smoke.json` records a guarded build:
  - the measured configuration passes, with smoke numbers identical to the 09-25 run;
  - all four unsupported combinations tested are rejected.

## License

b12x is licensed under the Apache License 2.0 (`LICENSE` in the b12x tree;
`pyproject.toml` declares `license = "Apache-2.0"`). These patches modify b12x source files,
and the upstream license, Apache License 2.0, applies to them. A copy is included as
`LICENSE.b12x`. The ShapleyMCG License in the repository root covers this repository's own
scripts and documents, not these patches.
