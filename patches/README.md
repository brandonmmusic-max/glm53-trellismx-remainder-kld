# Kernel patches (b12x)

Two unified diffs against the b12x source tree shipped in the serving image
`verdictai/trellismx@sha256:ca6b80188dce154b91f49108b7d87792d2ba6328935afc71b44d1c0e6f6a1adf`.
The patches are relative to that tree as extracted from the image. The 16 files they touch
are byte-identical to the same paths at commit
`d564f6ca54c092497ec5ae7e07a272f55eec7dbe` of the TrellisMX b12x integration branch.

| Patch | Tree it produces | Used by | Files |
|---|---|---|---|
| `b12x-zero-remainder.diff` | `b12x-scratch` | Test A (zero-remainder timing arm) and the exploratory FC1/FC2 split | 5 modified |
| `b12x-dx2-rowpack.diff` | `b12x-dx2` | Amendment 2 (plane-based D-x2: K2, A2, exploratory split) and Amendments 3-4 (row-packed D-x2-RP: K3, A3 and the production-window `rp` arm) | 15 modified, 1 new (`p8_down_remainder.py`) |

Apply from the root of the extracted b12x tree:

```
patch -p1 < b12x-dx2-rowpack.diff
```

The diffs were produced with
`diff -ruN -x __pycache__ -x .git -x build -x '*.egg-info' a/ b/`.

## Regenerating the trees

The builder scripts regenerate these trees from the image's b12x source:
`../scripts/build_zero_remainder_b12x.py` builds `b12x-scratch`, and
`../scripts/build_dx2_b12x.py` builds `b12x-dx2`. First set `SRC` and `DST` in each
script. Every text replacement is counted, and a build aborts if the source does not match.
We checked both routes:
- running the builders on the image tree reproduces the measured trees file for file;
- applying these patches to a clean copy of the image tree does the same.

## Gating

All new kernel code sits behind `cutlass.const_expr` flags, which are set only on new
subclasses:
- `p8_zero_remainder` and `p8_zr_unmask` on `...ZR` / `...ZRU`;
- `p8_down_remainder` on `...DX2`;
- `p8_dx2_rowpack` on `...RP`.

With a flag off, the kernel traces exactly as before. The host-side changes do nothing unless
`B12X_P8_ZERO_REMAINDER` or `B12X_P8_DOWN_REMAINDER` is set when the MoE runtime is
constructed, and the scratch layouts default to one plane.

Flag-off behaviour was checked on real outputs:
- **D-x2 build.** With the flag off, outputs were bit-identical to the image's own kernels
  (layers 3 and 8, M16 and M3072, all four TP ranks summed; K2 in `../RESULTS.md`).
- **D-x2-RP.** Only the M <= 16 direct paths are row-packed. On the M64 and M3072 paths, RP
  outputs were bit-identical to prod (K3).
- **Zero-remainder arm.** Its outputs were bit-identical to the flag-off baseline in every
  Test A cell (the extra MMA adds exactly +0), and the unmasked positive control changed
  the outputs.

## License

b12x is licensed under the Apache License 2.0 (`LICENSE` in the b12x tree;
`pyproject.toml` declares `license = "Apache-2.0"`). These patches modify b12x source files,
and the upstream license, Apache License 2.0, applies to them. A copy is included as
`LICENSE.b12x`. The ShapleyMCG License in the repository root covers this repository's own
scripts and documents, not these patches.
