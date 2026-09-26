# Scripts

These are the scripts exactly as they ran, plus one summary script added for this repository
(`summarize_rowpack_timing.py`, marked below). Local paths are replaced by placeholders:
`<workspace>` (the directory that held the working repositories), `<home>`,
`<data-volume>` (captures, BF16 source shards, teacher logits) and `<model-volume>`
(the TrellisMX r27 checkpoint). Set these to your own locations before running anything.
They are research harnesses, not a package. Most of them run inside the serving image
with a patched b12x tree bind-mounted over `/opt/glm53-flash/b12x`, and the layer-level
harnesses import modules from the campaign repository
(`glm53-hadamard-shapleymcg-kld`, see the top-level README).

In the harnesses, `/work` is this project's root mounted into the container and
`/checkpoint` is the TrellisMX r27 checkpoint.

## Kernel builds

| Script | Purpose |
|---|---|
| `build_zero_remainder_b12x.py` | Builds the `b12x-scratch` tree for Test A: adds `...ZR` and `...ZRU` kernel subclasses, selected at compile time by `B12X_P8_ZERO_REMAINDER` (1 both GEMMs, 2 both positive control, 3 FC2 only, 4 FC1 only, 5 FC2 positive control). Every replacement is counted and the build aborts on a mismatch. |
| `build_dx2_b12x.py` | Builds the `b12x-dx2` tree for Amendments 2 and 3: the plane-based down-hop remainder (`...DX2` classes, `B12X_P8_DOWN_REMAINDER=1`) and the row-packed variant (`...RP` classes, `B12X_P8_DOWN_REMAINDER=rp`). This version built the tree measured in K2/K3, A2/A3 and both production windows. |
| `build_dx2_rp2_b12x.py` | The later version of the same builder, which adds the both-hop row-pack D-x2-RP2 (`...RP2` FC1 classes and the gated input prologue, `B12X_P8_DOWN_REMAINDER=rp2`; Amendment 6). It writes to `DX2_DST`, and it built the `b12x-dx2rp2` tree measured in the RP2 smoke test, K4, A4 and the Amendment 7 `rp2-fp8` arm. |

The builders copy the image's b12x source (`SRC`) and write the patched tree to `DST`
(`DX2_DST` for the RP2 builder). They delete the destination first, so point it at a scratch
directory. `../patches/` holds the resulting diffs.

## Test A: zero-remainder timing (PREREG)

| Script | Purpose | Output |
|---|---|---|
| `run_timing.sh` | Docker launcher: serving image, `b12x-scratch` mounted read-only, no network, one GPU, compile caches off. | |
| `timing_harness.py` | Builds `P8NativeTPMoE` for TP rank 0 of one routed layer the way the vLLM TrellisMX MoE method does, feeds real routed-block inputs and top-8 routes from the fit capture, checks `zr == base` exactly and `zru != base`, then times CUDA-graph replays in interleaved blocks. | `results/timing_raw.json` |
| `analyze_timing.py` | Paired per-block ratio, median, and a percentile bootstrap CI of the median (20,000 resamples, seed 20260923). Applies the decision rule. | `results/timing_analysis.json` |
| `run_timing_split.sh`, `timing_split_harness.py`, `analyze_timing_split.py` | Exploratory (after the FAIL): splits the extra MMA by GEMM (FC1 vs FC2). | `results/timing_split_raw.json`, `results/timing_split_analysis.json` |

The GPU clock checks (`results/gpucheck_*.json`) use the same harness via `run_timing.sh`,
with one M1 cell on layer 8, one GPU at a time.

## Test B and B2: layer-level numerics (PREREG, Amendment 2)

| Script | Purpose | Output |
|---|---|---|
| `fetch_fit_capture.py` | Downloads only the roles-v3 `fit` windows of the chosen layers' routed-block captures from the teacher-logits dataset (pinned capture revision). Each window goes to its true byte offset in a sparse file, and every range is hashed. | local capture tree |
| `remainder_damage.py` | Test B / B2 harness. It reference-decodes the r27 sidecars, runs the coupled reference forward and measures routed-output damage against the BF16 source experts, with selectable activation carriers (`a8`, `a8x2`, `exact`) at each hop. It asserts that the production arm is bit-identical to `coupled_expert_reference`. | `results/testB/layer-*.json`, `results/testB2/layer-*.json` |
| `run_testB.py` | Runs Test B layer by layer as the inputs land, then runs the like-for-like screen check. | |
| `screen_repro.py` | Reruns the 09-04 activation screen's ordinary-E4M3 arm on its frozen plan and adds a two-term arm. | `results/testB/screen-repro.json` |
| `run_testB2.py` | Runs B2 on the fresh layers after the A2 timing run has finished. | |
| `analyze_BC.py` | Tests B and C: pooled geometric-mean ratio with a paired window BCa bootstrap (its own BCa implementation with jackknife acceleration, 20,000 replicates, seed 20260923). | `results/analysis_BC.json` |
| `analyze_B2_A2.py` | Amendment 2 analysis: B2 (same BCa) and A2 timing medians. | `results/analysis_amendment2.json` |

## Test C: H512 before the NVFP4 MLA latent record (PREREG, Amendment 1)

| Script | Purpose | Output |
|---|---|---|
| `kv_hadamard.py` | Builds stand-in MLA latents and absorbed queries from the NVFP4 carrier's attention weights. Quantizes the latents with a torch mirror of the two-level NVFP4 record, unrotated, with H512, and with H512 plus random signs. Measures latent and attention-logit error. | `results/testC/A*-C*.json` |
| `run_testC.py` | Runs the decisive and sensitivity configurations as their captures land. | |
| `validate_kv_mirror_host.py`, `validate_kv_mirror_image.py` | Check the torch mirror against the real `concat_and_cache_glm_next_mla` writer inside the image (dump, write, dequantize, compare). | `.pt` intermediates, not included |

## Amendments 2, 3 and 6: real-kernel closure and timing

| Script | Purpose | Output |
|---|---|---|
| `run_dx2.sh`, `dx2_smoke.py` | Smoke test: every M path compiles and runs deterministically, and D-x2 differs from prod by an activation-error-sized amount. | |
| `run_rp2_smoke.sh`, `rp2_smoke.py` | RP2 smoke test with the `b12x-dx2rp2` tree mounted. Every M path compiles and runs. RP2 is compared with prod and with RP on the direct paths, and checked bit-identical to prod on the M > 16 paths. | console output |
| `dx2_base_outputs.py` | K2 input: the Test B fit tokens and their BF16-source routed outputs. | `results/k2/base-layer-*.pt`, not included |
| `run_k2.sh`, `dx2_closure.py` | Closure. All four TP4 rank sidecars run one at a time and their outputs are summed. Paths M1, M16, M64 and M3072. Arms: K2 `K2_ARMS=prod,dx2`, K3 `K2_ARMS=prod,rp K2_BUILD=rp`, K4 `B12X_TREE=b12x-dx2rp2 K2_ARMS=prod,rp2 K2_BUILD=rp2`. | `results/k2/closure-dx2build.json`, `closure-rp.json`, `closure-rp2.json` |
| `run_k2_image.sh` | Same closure harness on the image's own b12x (no mount), used for the flag-off bit-identity check. | `results/k2/closure-image.json` |
| `run_a2.sh`, `dx2_timing.py` | Real-kernel timing. A2 uses `A2_ARM=1`. A3 uses `A2_ARM=rp A2_MS=1,4,16`. A4 uses `B12X_TREE=b12x-dx2rp2 A2_ARM=rp2 A2_MS=1,4,16`. | `results/a2_timing_raw.json`, `a3_timing_raw.json`, `a4_timing_raw.json` |
| `summarize_rowpack_timing.py` | Added for this repository. Summarizes an A3 or A4 raw timing file with the A2 estimator from `analyze_B2_A2.py`: median of paired per-block ratios, percentile bootstrap CI (20,000 resamples, seed 20260923), and the decision rule. It reproduces the A3 and A4 numbers in `RESULTS.md`. | `results/a4_timing_summary.json` |
| `run_dx2_split.sh`, `dx2_split_timing.py` | Exploratory (after the A2 FAIL): splits the D-x2 cost between the FC1 epilogue and FC2. | `results/dx2_split_timing_raw.json` |

`dx2_closure.py`, `dx2_timing.py`, `run_k2.sh` and `run_a2.sh` are the versions that added
the `rp2` arm and the `B12X_TREE` switch. They are backward-compatible with the earlier
arms:
- `B12X_TREE` defaults to `b12x-dx2`;
- the `1` and `rp` arms select the same flags as before;
- the new `p8_input_rowpack` check reads `False` on trees without RP2.

## Amendments 4, 5 and 7: production windows (end-to-end KLD and speed)

| Script | Purpose | Output |
|---|---|---|
| `window_nvfp4_20260925.py` | NVFP4-KV window. Drains and stops production, then captures and scores the 32 conditional-fit windows with a fresh server for `control` and then `rp`, runs the paired BCa analysis, measures descriptive speed on a private port, and always restores production. It uses `p8_decode_protocol` from the campaign repository. | `results/kld-rp-20260925/` |
| `window_fp8_20260925.py` | FP8-KV window. Same procedure with `KV_CACHE_DTYPE=fp8`: `control-fp8` then `rp-fp8`, plus `rp-fp8` speed. It also records each capture server's KV capacity in its runtime audit. | `results/kld-fp8-20260925/` |
| `build_inputs_128.py` | Builds the verified input list for all 128 conditional-fit windows. It uses the same record format and hash checks as the 32-window runs; the original 32 keep their order and records. | `results/kld-cf128/verified-inputs.json` |
| `window_cf128_20260925.py` | Final 128-window run (Amendment 7), all arms with FP8 MLA KV: `control-fp8`, `rp2-fp8` (with the `b12x-dx2rp2` tree), then `tr3-fp8` (the 09-09 TR3 attempt-2 capture launch). It runs the paired analyses, then an `rp2-fp8` speed arm. The run is in progress. | pending |

The window scripts also operate the serving host: systemd user unit, lock files and
container names. That part is specific to our machine and is kept only as a record of what
ran.
