# FP8 remainder carriers and NVFP4-KV rotation on TrellisMX P8 kernels (GLM-5.3-Flash)

Brandon M. Music, September 2026

This repository records a preregistered series of kernel and numerics tests on our
TrellisMX r27 P8 kernels for GLM-5.3-Flash. The series runs from layer-level gates to
end-to-end true-decode KLD measurements taken in two production windows, one with NVFP4 MLA
KV and one with FP8 MLA KV. The repository contains:
- the preregistration (`PREREG.md`), with every amendment written before its numbers;
- the running results log (`RESULTS.md`);
- every script that produced a number;
- the raw and analysed result files;
- the b12x kernel patches.

**Status (2026-09-25).** Tests A-C and Amendments 2-6 are complete. That covers the FP8-KV
production window and the gates for the both-hop row-pack D-x2-RP2, both of which passed. The
final 128-window run (Amendment 7) is running; see
[In progress / pending](#in-progress--pending).

## Summary

| Question | Test | Outcome |
|---|---|---|
| Is a second FP8 MMA per decoded weight fragment free at decode? | A: zero-remainder timing | **No (FAIL).** Median time ratio 1.0698 / 1.0854 at M1 and 1.1650 / 1.1484 at M4 (layers 8 / 3); the bar was <= 1.03. |
| How much routed-output damage does a two-term (hi + lo) FP8 activation carrier remove? | B: layer level, 6 layers | Pooled damage ratio 0.490 (-51.0%), BCa 95% CI [0.427, 0.562]; 6/6 layers improve. |
| Does a fixed H512 rotation before the NVFP4 MLA latent record reduce error? | C: 6 MLA layers | **No (FAIL).** Pooled ratio 1.0011 for latent NMSE and 1.0055 for attention-logit error. |
| Down-hop-only remainder in a second data plane (D-x2) | K2 / A2 / B2 | Closure PASS. Timing FAIL: 1.071-1.079 at M4, 1.163-1.187 in prefill. Fresh-layer damage 0.744 (-25.6%), B2 PASS. |
| Down-hop remainder packed into the idle MMA row +8 (D-x2-RP) | K3 / A3 | Closure PASS. Timing PASS: median ratio <= 1.0272 at M1 and M4. |
| Row-packed remainder at both hops (D-x2-RP2) | K4 / A4 | Closure PASS. Timing PASS: median ratio <= 1.0273 at M1 and M4. |
| End-to-end true-decode KLD with NVFP4 MLA KV | Amendment 4 | control 0.0350129 vs rp 0.0330631: -0.00195 (-5.57%), paired BCa 95% [-0.005975, +0.000118]; rp lower in 19/32 windows. Preregistered reading: **no detectable change**. Decode speed was not slower. |
| End-to-end true-decode KLD with FP8 MLA KV | Amendment 5 | control-fp8 0.0311958 vs rp-fp8 0.0309337: -0.00026 (-0.84%), paired BCa 95% [-0.001146, +0.001246]; rp lower in 19/32: **no detectable change**. FP8 vs NVFP4 KV for the control arm (cross-run): -10.9% [-0.00898, -0.00171], lower in 24/32. |

**Reading.**
- **FP8 MLA KV** is the one significant end-to-end gain in this series: about -11% KLD
  against NVFP4 KV. It costs KV capacity: -35% tokens in the production profile.
- **The row-packed remainder's end-to-end effect is small and not established.** It is
  -5.6% under NVFP4 KV and -0.8% under FP8 KV, and both paired intervals cross 0.

## Background and motivation

### The P8 kernels

In the TrellisMX r27 checkpoint, the routed experts of GLM-5.3-Flash's 42 MoE layers are
trellis bitstreams at K4 or K5 bits per weight, chosen per layer. All other weights come
from an NVFP4 carrier checkpoint.

The P8 kernels (in b12x) work as follows:
- **Weights.** Each weight fragment is decoded in registers to E4M3 with a procedural MCG
  codebook, whose constants and state construction are ported from ExLlamaV3's procedural
  MCG decoder (see [Background](#background)). The decoded fragment goes straight into a
  block-scaled tensor-core MMA (`mma.sync`, `kind::mxf8f6f4`, m16n8k32).
- **Activations.** E4M3 with a power-of-two UE8M0 scale per 32 elements along K
  ("E4M3/UE8M0-K32"). They are quantized at two hops: the FC1 (gate/up) input, and the FC2
  (down) input after SwiGLU.
- **Coupled transform.** The checkpoint's block Hadamards and per-channel scales sit around
  both GEMMs.
- **Kernel paths by routed rows per layer (M).** M <= 16 (decode) uses "direct" small-M
  kernels, M 17-128 uses grouped M32 tiles, and larger M uses the prefill kernels.

### Why a remainder (two-term) FP8 carrier

E4M3 keeps 3 mantissa bits. The two-term carrier sends `hi = q(x)` and
`lo = q(x - dq(hi))`, each with its own UE8M0 scale. In a kernel this is a second MMA on
the same decoded weight fragment, accumulating into the same FP32 accumulator, because
`x W ~= hi W + lo W`.

Two observations motivated it:
1. **Activation error is a large share of the damage.** With exact BF16 weights and
   production activations, the routed-output damage is 0.132-0.589 of production's on the
   Test B layers (the "act. share" column below).
2. **The expensive step is decoding the weight fragment.** On a trellis kernel, reusing an
   already decoded fragment for a second MMA might be close to free at small M.

Test A measured (2) directly. Its second MMA uses an A operand that is really loaded and
then zeroed at runtime, so outputs are unchanged and only the cost shows. Test B measured
(1) at layer level.

### Why row-packing is free on one-route decode tiles

On the direct decode paths (M <= 16), every physical 16/32-row MMA tile holds exactly one
route row (`valid_rows == 1`). The other rows of the A tile are zeros that the tensor core
multiplies anyway. In the m16n8 accumulator fragment, each thread holds the same columns
for rows q and q+8.

D-x2-RP uses that spare row:
- **FC1 epilogue.** Writes the down-input remainder `lo = q(v - dq(hi))`, with its own
  UE8M0 byte, into row +8 of the route's own tile. This is the same math as the
  plane-based D-x2.
- **FC2 main loop.** Unchanged. It already stages and multiplies row 8, and its odd lanes
  already supply the row-8 A scale.
- **FC2 epilogue.** Folds row 8 into row 0 in registers before the existing FP16 store
  (`fragment[0] += fragment[2]`, `fragment[1] += fragment[3]`).

There is no extra MMA, no extra staging and no second plane. Those are exactly what the
plane-based D-x2 paid for when it failed A2. Paths with M > 16 keep the production
single-term kernels, so the variant affects decode only, and only a decode-scored KLD can see
its effect.

### Why the KV rotation was tested

Production stores the MLA latent cache (512 dims, no RoPE payload) as a two-level NVFP4
record:
- a per-token scale `s_t = amax_t / (6 * 448)`;
- E4M3 scale bytes per 16 elements, relative to `s_t`;
- E2M1 values.

A fixed orthogonal rotation before low-bit quantization spreads outlier channels across the
vector, as in QuaRot. Here its inverse could be folded into the absorbed `W_UK` / `W_UV`, so
it would cost nothing at runtime. Test C asked whether there are outliers to spread. There
are not: after `kv_a_layernorm`, the per-token amax/RMS is 3.25, the value for a 512-dim
Gaussian. The NVFP4 error (NMSE 0.0090) is already the Gaussian floor.

## Preregistration discipline

`PREREG.md` was written on 2026-09-23, before any number for Tests A-C existed. It fixes
each test's harness, metric, uncertainty estimator and decision rule. Every later amendment
was written before any number it governs and says what was already known at the time.

| Part | Written | Content |
|---|---|---|
| Original | 2026-09-23 | Tests A (timing), B (layer numerics) and C (KV rotation), with decision rules. |
| Amendment 1 | 2026-09-23, before any Test C number | Moves Test C to the model's actual MLA layers. Only every fourth layer is DeepSeek-sparse-attention MLA; the rest are KDA linear attention. The residual is four hyper-connection streams. Records that Test A had already failed. |
| Amendment 2 | 2026-09-23, before any of its numbers | Confirms the post-hoc down-hop signal on fresh layers (B2) and on a real kernel: closure (K2) and timing (A2). |
| Amendment 3 | 2026-09-23, before any D-x2-RP number | Row-packed down-hop remainder: closure (K3) and timing (A3). |
| Amendment 4 | 2026-09-25, before any end-to-end number | Decode-scored KLD with the 09-09 reference protocol, plus descriptive speed, in an approved production window. |
| Amendment 5 | 2026-09-25, before any number | FP8-KV window, then the FC1-input row-pack. |
| Amendment 6 | 2026-09-25, before any D-x2-RP2 number | Gates for the both-hop row-pack (D-x2-RP2): closure (K4) and timing (A4). |
| Amendment 7 | 2026-09-25, before any 128-window number | Final run on all 128 conditional-fit windows with FP8 MLA KV. Arms: `control-fp8`, `rp2-fp8` and `tr3-fp8` (TR3 4bpw). Primaries are `rp2 - control` and `rp2 - TR3`. There is also an `rp2-fp8` speed arm. |

Rules we kept:
- **Fixed rules.** Decision thresholds and estimators are set in advance. Exploratory
  analyses are labelled as such and cannot change a decision. For example, Test B's FC1-only
  and down-only arms were added after Test A and are reported only.
- **A preregistered FAIL stops its direction.** Test A's FAIL stopped the two-hop design
  whatever Test B showed. Work on the down hop restarted under Amendment 2, on fresh data.
- **Protected data roles.**
  - The layer tests read only the roles-v3 `fit` windows.
  - The KLD runs in Amendments 4 and 5 use 32 `conditional-fit` windows that earlier
    development measurements had already opened.
  - Amendment 7 widens this to all 128 conditional-fit windows; 96 of them are scored for the
    first time.
  - Selection, confirmation and final windows stay unopened.

| Gate | PASS requires |
|---|---|
| Test A | Median paired time ratio <= 1.03 at M1 and M4 on both layers, with every output exact. |
| Test B | Pooled geometric-mean damage reduction >= 5%, >= 5 of 6 layers improve, and the pooled log-ratio CI excludes 0. |
| Test C | Both metrics drop >= 20% pooled, and the CI excludes 0. |
| B2 | Pooled down-only reduction >= 20%, >= 5 of 6 layers improve, and the upper CI bound of the pooled log ratio < 0. |
| K2 / K3 | Per layer and path: `abs(D_kernel(prod) / D_ref(P-A8) - 1) <= 5%`, and the arm/prod ratio within +/-0.05 of the reference down-only ratio. K3 also requires M64/M3072 outputs bit-identical to prod. |
| A2 | Decode: median <= 1.03 at M1 and M4. Prefill: median <= 1.05 at M512 and M2048. |
| A3 | Median <= 1.03 at M1 and M4 on both layers. |
| Amendments 4 and 5 reading | Improvement if the paired mean < 0 and the CI excludes 0. No detectable change if the CI includes 0. Harm if the mean > 0 and the CI excludes 0. |
| K4 (Amendment 6) | As K3, with `D_kernel(RP2)/D_kernel(prod)` within +/-0.05 of the reference both-hop ratio `D(P-A8x2)/D(P-A8)`, and M64/M3072 bit-identical to prod. |
| A4 (Amendment 6) | Median `RP2/prod` <= 1.03 at M1 and M4 on both layers. |
| Amendment 7 primaries | (1) `rp2-fp8 - control-fp8`: improvement if the mean < 0 and the CI excludes 0. (2) `rp2-fp8 - tr3-fp8`, reported three ways: whether the CI includes 0, whether the whole CI lies within +/-5% of TR3's mean, and the same at +/-10%. |

## Methods

### Model, checkpoint, runtime, hardware

- **Base model.** `zai-org/GLM-5.3-Flash-BF16`. It is the KLD teacher and the reference for
  the layer-level damage measurements.
- **Checkpoint.** TrellisMX r27 (`brandonmusic/GLM-5.3-Flash-TrellisMX-MXFP8`): per-rank
  trellis sidecars for the routed experts (four TP ranks per layer), overlaid on the NVFP4
  carrier `local-inference-lab/GLM-5.3-Flash-NVFP4`.
- **Runtime.** Serving image `verdictai/trellismx@sha256:ca6b8018...` (full digest under
  [Sources](#sources)). It is vLLM-based; the server reports
  `0.26.1rc0+glm53.flash.nvfp4.luke.clean.r1.vllme75bcfd.b12x58a046f`. The b12x kernels live
  under `/opt/glm53-flash/b12x`, and every patched arm bind-mounts a patched b12x tree over
  that path.
- **Hardware.** Four NVIDIA RTX PRO 6000 Blackwell GPUs (two Max-Q, two Workstation
  Edition), PCIe, TP4, with a 300 W limit each (1200 W total, recorded in the benchmark
  JSONs).

### Layer-level numerics (Tests B and B2)

- **Method.** `scripts/remainder_damage.py` implements the campaign's `p8_layer_rate_damage`
  method, unchanged except for the activation arms:
  - the r27 rank sidecars are reference-decoded with the kernel's procedural-MCG E4M3
    codebook;
  - the exact coupled reference forward runs over all 288 experts;
  - routed-output damage is
    `D = sum_t ||S(t)||^2 / sum_t ||routed_output(t)||^2` against the BF16 source experts.
- **Tokens.** 64 fit windows x 48 linspace tokens = 3,072 tokens per layer. Routed-block
  inputs and top-8 routes come from the BF16 capture in the teacher dataset.
- **Quantizer.** `q` is E4M3 with a UE8M0 amax scale per 32-element block (the campaign's
  `_qdq_e4m3_k32`). The two-term carrier is `hi + lo` in FP32, which is what two FP8 MMAs
  accumulate.

  | Arm | Weights | FC1 input | Down input |
  |---|---|---|---|
  | `P-A8` (production) | P8 | single-term | single-term |
  | `P-A8x2` | P8 | two-term | two-term |
  | `P-Aexact` (weight-only error) | P8 | exact | exact |
  | `B-A8`, `B-A8x2` (activation-only error) | BF16 in coupled coordinates | single / two-term | single / two-term |
  | `B-Aexact` (closure) | BF16 in coupled coordinates | exact | exact |
  | `P-A8x2-fc1only`, `P-A8x2-downonly` (added after Test A, reported only) | P8 | two-term / single-term | single-term / two-term |

- **Checks.** The harness asserts that `P-A8` is bit-identical to the campaign's
  `coupled_expert_reference`. `B-Aexact` closes at 4-9e-8 relative.
- **Uncertainty.** Paired window BCa bootstrap: one resample of the 64 window indices is
  applied to every layer at once. The statistic is the mean over layers of
  `log(sum_w D_new / sum_w D_ref)`, and its exponential is the pooled geometric-mean ratio.
  20,000 replicates, seed 20260923, jackknife acceleration (`scripts/analyze_BC.py`).
- **Like-for-like check.** We reran the 09-04 activation screen's ordinary-E4M3 arm on its
  frozen plan (layer 3, its 16 experts, fit role, domain-balanced, offset 32, count 32).

### KV rotation (Test C)

- **Stand-in latents.** Attention inputs are not in the capture. The stand-in for MLA layer A
  comes from the routed-block capture of layer C:
  - input: `a = gamma_in(A) * (m_C / gamma_post(C))`;
  - latent: `c = kv_a_layernorm(kv_a_proj(a))`, using the carrier's BF16 attention weights;
  - queries: `q_a_proj -> q_a_layernorm -> q_b_proj`, absorbed with each head's `W_UK`.
- **Layers.** The decisive pairs A:C are 7:6, 15:14, 19:18, 27:26, 35:34 and 43:42. The
  sensitivity runs use 3:3, 23:23 and 43:43.
- **Tokens.** Keys are all 2,048 tokens of each fit window. Queries are the 48 Test B
  positions per window, over causal pairs.
- **Quantizer.** A torch mirror of b12x's two-level record writer (`kv_cache.py`,
  `per_token_scale=True`), checked against the real `concat_and_cache_glm_next_mla` writer.
- **Arms.** Unrotated, normalized Sylvester H512 (decisive), and H512 with fixed random
  signs (reported only).
- **Metrics.** Latent NMSE, and relative attention-logit error
  `sum (q_abs . (c_hat - c))^2 / sum (q_abs . c)^2`. Both use the same paired window BCa as
  Test B.

### Real-kernel closure and timing (Test A, K2-K4, A2-A4)

K4 and A4 (Amendment 6) use the same harnesses as K3 and A3. The differences are the RP2
tree (`b12x-dx2rp2`, via `B12X_TREE`) and the `rp2` arm (`B12X_P8_DOWN_REMAINDER=rp2`).

- **Closure** (`scripts/dx2_closure.py`). Layers 3 (K5) and 8 (K4).
  - All four TP4 rank sidecars run one at a time, and their outputs are summed, which is
    what the all-reduce computes.
  - Tokens are the Test B fit tokens, fed in chunks that exercise every path: M1 (first 96
    tokens), M16, M64 (grouped M32) and M3072 (prefill).
  - Damage is `D_kernel = sum_t ||kernel_t - base_t||^2` against the BF16-source routed
    output.
- **Timing** (`scripts/timing_harness.py`, `scripts/dx2_timing.py`).
  - `P8NativeTPMoE` is built for TP rank 0 the way the vLLM TrellisMX MoE method builds it,
    with real routed-block inputs and top-8 routes.
  - CUDA-graph replay, blocks interleaved in alternating order, 60 blocks x 200 replays
    per cell. Decode cells use 4 input sets (240 blocks) and prefill cells use 2
    (120 blocks). The exploratory D-x2 split used 40 blocks and 2 input sets per cell.
  - Metric: the median of per-block paired time ratios, with a percentile-bootstrap CI of
    the median over blocks (20,000 resamples, seed 20260923).
  - Compile caches were off for every timing run: they key on the compile spec, not on the
    arm.

### End-to-end KLD and speed (Amendments 4 and 5)

Both production windows use the same protocol. Amendment 4 ran it with NVFP4 MLA KV and
Amendment 5 with FP8 MLA KV. The description below is for Amendment 4; the differences for
Amendment 5 are listed at the end of this section.

- **Protocol.** The 09-09 reference protocol of the published TrellisMX reference KLD
  measurement, unchanged:
  - the same 32 conditional-fit windows and BF16 teacher logits, each file's sha256
    verified;
  - forced-token true decode with 2048 input tokens and 2047 captured rows;
  - row 0 (the prefill row) excluded, which leaves 2046 true-decode rows per window and
    65,472 per arm;
  - `KL(teacher || student)` computed on CPU in FP64 over the full 154,880-token
    vocabulary, and each arm scored as the mean of its 32 window means.
- **Capture server.** A capture-only derivative of the serving image (local image id
  `sha256:0405a1c0...`; the same capture image as the published 09-09 measurement):
  - TP4/DCP4, MTP off, one sequence, NVFP4 MLA KV;
  - 4096 batched tokens, GPU memory utilization 0.97, max length 1M;
  - prefix caching on, with every prefix-cache hit counter required to stay 0.
- **Arms.** One fresh server each, in fixed order: `control`, then `rp`.
  - `rp` is the same launch plus the `b12x-dx2` tree mounted read-only and
    `B12X_P8_DOWN_REMAINDER=rp`, with its own JIT cache volume.
  - The `rp` server logged 168 `P8_DX2_ROWPACK_ACTIVE` lines (42 layers x 4 ranks);
    `control` logged 0.
  - With MTP off and one sequence, every decode step takes the M1 direct path, where RP is
    active.
- **Estimator.** Paired per-window differences `rp - control`, with a BCa 95% interval from
  `scipy.stats.bootstrap` (`method="BCa"`, 20,000 resamples, `random_state=20260902`). This
  is the estimator behind the published intervals. Each arm's mean also gets a window BCa
  interval.
- **Speed.** Descriptive only, not a gate.
  - Server: the production serving config (MTP with 3 speculative tokens, NVFP4 KV, 48
    sequences, 8192 batched tokens, GPU memory utilization 0.88) on a private port,
    `control` first.
  - Tool: `llm_decode_bench.py` 0.4.29 with `--skip-prefill`.
  - Cells: contexts 0 and 8192, concurrency 1 and 4, 30 s each, 2048 max tokens.
  - Before each cell: >= 90 s idle and all GPUs <= 55 C for 30 s.
  - One run per cell.
- **Amendment 5 (FP8 MLA KV).**
  - Both servers are launched with `KV_CACHE_DTYPE=fp8`. The arms run in fixed order:
    `control-fp8`, then `rp-fp8`.
  - The `rp-fp8` server logged 168 `P8_DX2_ROWPACK_ACTIVE` lines, and `control-fp8` logged 0.
  - The primary is the paired `rp-fp8 - control-fp8`, with the same estimator.
  - Speed was measured for `rp-fp8` only, with the same cells and cooling gate.
  - KV capacity is read from each server's engine startup log.
  - The window script (`scripts/window_fp8_20260925.py`) computes only the primary.
  - The cross-run comparisons pair the same 32 windows, using the same estimator and seed.
    They compare against the Amendment 4 arms and the published 09-09 values. Their inputs are
    the per-window records in `results/kld-rp-20260925/analysis.json`,
    `results/kld-fp8-20260925/analysis.json`, and the published 09-09 `comparison.json` files.

### Estimators and seeds

| Analysis | Estimator | Resamples | Seed |
|---|---|---|---|
| Test A, exploratory split, A2, A3 and A4 timing | percentile bootstrap of the median over blocks | 20,000 | 20260923 |
| Tests B, C and B2 | paired window BCa with jackknife acceleration | 20,000 | 20260923 |
| Test C random-sign arm | fixed random signs | n/a | 20260923 |
| End-to-end KLD, both windows, including cross-run comparisons | `scipy.stats.bootstrap`, BCa, over window-paired differences | 20,000 | 20260902 |

## Results

Every number below comes from `RESULTS.md` or the files under `results/` and is copied at
the source's precision.

### Test A: zero-remainder timing (preregistered). FAIL

**Setup.** 60 interleaved blocks x 200 replays x 4 input sets per cell, on GPU 0.

**Checks.** Every output was bit-identical to the baseline, the baseline was deterministic,
and the unmasked positive control changed the outputs.

Median time ratio (arm Z / baseline) with 95% CI (`results/timing_analysis.json`):

| layer | M1 | M4 | M16 (reported only) |
|---|---|---|---|
| 8 (K4) | 1.0698 [1.0697, 1.0699] | 1.1650 [1.1648, 1.1653] | 1.1341 |
| 3 (K5) | 1.0854 [1.0853, 1.0855] | 1.1484 [1.1406, 1.1553] | 1.1155 |

**Exploratory (after the FAIL): the extra MMA split by GEMM.** Run on GPU 2. All arms were
exact, and the FC2 positive control changed the outputs (`results/timing_split_analysis.json`).

| layer | M | both | FC2 (down) only | FC1 (gate/up) only |
|---|---|---|---|---|
| 8 K4 | 1 | 1.0786 | 1.0244 | 1.0779 |
| 8 K4 | 4 | 1.1741 | 1.0247 | 1.1675 |
| 8 K4 | 16 | 1.1498 | 1.0229 | 1.1296 |
| 3 K5 | 1 | 1.0758 | 1.0009 | 1.0748 |
| 3 K5 | 4 | 1.1532 | 1.0240 | 1.1228 |
| 3 K5 | 16 | 1.1189 | 1.0186 | 1.0980 |

The zero-remainder arm is a lower bound on a real remainder's cost. It leaves out the lo-term
quantize in the FC1 epilogue, the lo-scale loads, and staging the lo tile.

**Side finding: GPU 0 clock pin.** Test A ran while GPU 0 held a stale clock pin (flat
1642 MHz). Its M1 kernel took 108.5 us, against 77.8 us on the twin Max-Q GPU 2. The pin was
reset (`nvidia-smi -rgc`) before any Amendment 2 run. The exploratory split on GPU 2 gave
ratios similar to Test A's (`results/gpucheck_*.json`).

### Test B: two-term carrier, layer level (preregistered). Criteria met

**Pooled D(P-A8x2)/D(P-A8) = 0.490 (-51.0%), BCa 95% CI [0.427, 0.562], 6/6 layers improve**
(`results/analysis_BC.json`).

| layer | K | rel. damage (prod) | two-term | act. share | wt. share | FC1-only* | down-only* |
|---|---|---|---|---|---|---|---|
| 3 | 5 | 1.32e-3 | 0.549 | 0.447 | 0.548 | 0.930 | 0.618 |
| 8 | 4 | 9.72e-3 | 0.869 | 0.132 | 0.869 | 0.931 | 0.938 |
| 18 | 4 | 1.21e-3 | 0.474 | 0.516 | 0.474 | 0.951 | 0.517 |
| 23 | 4 | 1.74e-4 | 0.386 | 0.589 | 0.386 | 0.814 | 0.526 |
| 33 | 5 | 1.92e-4 | 0.287 | 0.408 | 0.287 | 0.649 | 0.512 |
| 43 | 5 | 3.58e-4 | 0.553 | 0.444 | 0.554 | 0.950 | 0.613 |

\* Arms added after Test A, reported only.
- Pooled down-only: 0.606 [0.549, 0.673].
- Pooled FC1-only: 0.863 [0.842, 0.909].

**Per domain** (two-term ratio): general 0.596, legal 0.509, code/agentic 0.625,
reasoning 0.403.

**Like-for-like check.** The 09-04 screen's ordinary-E4M3 geometric-mean NMSE reproduces as
7.3725278e-4 against the published 7.3725270e-4 (relative difference 1e-7). On the same
frozen plan, the two-term carrier gives 2.76e-7, and 16/16 experts improve
(`results/testB/screen-repro.json`).

Test B is a layer-level proxy, not KLD. Under the preregistration, Test A's failure stopped
the two-hop design regardless of this result.

### Test C: H512 before the NVFP4 MLA latent record (preregistered, amended). FAIL

**Mirror check.** On real stand-in latents, the torch mirror matches the real writer's
304-byte record on 99.96% of elements, and error energy agrees to 1.6e-5.

| metric | H512 pooled ratio | H512 x random signs |
|---|---|---|
| latent NMSE (unrotated = 0.0090) | 1.0011 [1.0009, 1.0013] | 1.0006 [1.0004, 1.0008] |
| attention-logit error | 1.0055 [1.0044, 1.0066] | 1.0056 [1.0044, 1.0068] |

The same-layer stand-ins (3:3, 23:23, 43:43) agree within +/-1.6%. The latent after
`kv_a_layernorm` is already Gaussian-shaped: per-token amax/RMS is 3.25, and there are no
outlier channels for a rotation to spread (`results/testC/`).

### Amendment 2: plane-based down-hop remainder (D-x2)

**K2, real-kernel closure: PASS** (`results/k2/closure-dx2build.json`; reference values
from `results/testB/`).

| layer | path | D_kernel(prod)/D_ref(P-A8) | D-x2/prod (kernel) | reference down-only |
|---|---|---|---|---|
| 3 | M1 (96 tok) | 1.0025 | 0.3650 | 0.3630 |
| 3 | M16/M64/M3072 | 1.0015 | 0.6189 | 0.6183 |
| 8 | M1 (96 tok) | 1.0008 | 0.7844 | 0.7843 |
| 8 | M16/M64/M3072 | 1.0003 | 0.9384 | 0.9384 |

With the flag off, the D-x2 build's outputs are bit-identical to the image's own kernels
(layers 3 and 8, M16 and M3072, four ranks summed).

**A2, real-kernel timing: FAIL** (decode and prefill). D-x2/prod medians
(`results/analysis_amendment2.json`):

| layer | M1 | M4 | M16 | M64 | M512 | M2048 |
|---|---|---|---|---|---|---|
| 8 K4 | 1.026 | **1.079** | 1.009 | 1.053 | **1.172** | **1.187** |
| 3 K5 | **1.030** | **1.071** | 1.038 | 1.047 | **1.163** | **1.185** |

Exploratory split (not preregistered): the cost is mostly FC2 (lo staging plus the extra
MMA), e.g. +15-16% at M512. The FC1 epilogue adds 0-3% (`results/dx2_split_timing_raw.json`).

**B2, fresh-layer numerics: PASS.** Pooled down-only D ratio **0.744 (-25.6%), BCa 95% CI
[0.673, 0.807], 6/6 improve**.

| layer | K | down-only | both hops | activation share |
|---|---|---|---|---|
| 6 | 4 | 0.939 | 0.865 | 0.14 |
| 14 | 4 | 0.901 | 0.840 | 0.16 |
| 26 | 4 | 0.896 | 0.851 | 0.15 |
| 21 | 5 | 0.804 | 0.802 | 0.23 |
| 34 | 5 | 0.747 | 0.719 | 0.29 |
| 42 | 5 | 0.372 | 0.282 | 0.72 |

The fresh-layer gain is smaller than on the Test B layers (post-hoc down-only 0.606): the K4
layers with high weight error gain little.

### Amendment 3: row-packed down-hop remainder (D-x2-RP)

**K3, closure: PASS** (`results/k2/closure-rp.json`).
- M1/M16 damage ratios against the reference:
  - layer 3: 0.3650 / 0.6189 (reference 0.3630 / 0.6183);
  - layer 8: 0.7844 / 0.9384 (reference 0.7843 / 0.9384).
- These match the plane-based D-x2 to FP32 rounding.
- M64 and M3072 outputs are bit-identical to prod.

**A3, timing: PASS.** RP/prod medians over 240 blocks (`results/a3_timing_raw.json`, same
estimator as A2):

| layer | M1 | M4 | M16 |
|---|---|---|---|
| 8 K4 | 1.0254 | 1.0200 | 0.9350 |
| 3 K5 | 1.0009 | 1.0272 [1.016, 1.039] | 1.0030 |

K3, A3 and B2 all passed; B2's down-only arm is the identical math. Under Amendment 3, that
justified asking for a production window for a decode-scored end-to-end KLD run.

### Amendment 4: end-to-end true-decode KLD and speed

The production window ran on 2026-09-25 from 19:00:43 to 20:00:59. Production was restored
and verified healthy afterwards (`results/kld-rp-20260925/restoration.json`).

| arm | mean true-decode KLD | window BCa95 |
|---|---|---|
| control | 0.0350129 | [0.02911, 0.04317] |
| rp (D-x2-RP) | 0.0330631 | [0.02801, 0.03983] |

**rp - control = -0.00195 (-5.57%), paired BCa95 [-0.005975, +0.000118], rp lower in 19/32
windows.** The preregistered reading is **no detectable change**: the CI crosses 0 by 0.0001
(`results/kld-rp-20260925/analysis.json`).

Post-hoc, reported only (65,472 rows):
- median row KL: -1.5%;
- q90 / q99 / q99.9: -5% / -5% / -8%;
- mean without the top 0.5% of rows: -4.0% (rows above the pooled q99.5 dropped from both
  arms);
- window medians: lower in 23/32.

Context from the published 09-09 measurements. These comparisons are cross-run, so each
carries server-run noise.
- The 09-25 control vs the published 09-09 control (0.0354562): -1.25% in the mean, and
  3.2% per-window SD.
- Paired against the published TR3 4bpw with the same NVFP4 KV (0.03048):
  - control: +0.00453 [+0.00029, +0.01032];
  - rp: +0.00258 [-0.00119, +0.00577].

  RP closes 43% of the gap.
- RP with NVFP4 KV is +3.5% vs TrellisMX 09-09 with FP8 KV (0.03195).

**Speed.** Production config with MTP3 on a private port. The cooling gate passed before
every cell, and each cell ran once. Values are aggregate decode tokens/s (`aggregate_tps`).
"C1 0K" means concurrency 1 at context 0 (`results/kld-rp-20260925/speed/`).

| cell | control | rp | MTP accept (control / rp) |
|---|---|---|---|
| C1 0K | 184.3 | 191.1 (+3.7%) | 0.436 / 0.553 |
| C4 0K | 300.1 | 301.2 (+0.4%) | 0.550 / 0.554 |
| C1 8K | 184.1 | 193.9 (+5.3%) | 0.538 / 0.610 |
| C4 8K | 300.7 | 299.0 (-0.6%) | 0.669 / 0.656 |

RP is not slower. The C1 gains track higher MTP acceptance in these runs. That could come
from more BF16-like target logits or from different generated text; single runs cannot tell
which.

### Amendment 5: FP8 MLA KV window

The second production window ran on 2026-09-25 from 20:29:49. Production was restored and
verified healthy at 21:17:30 (`results/kld-fp8-20260925/window.log`, `restoration.json`).
The protocol and windows are the same as Amendment 4, with `KV_CACHE_DTYPE=fp8`. The
`control-fp8` server logged 0 RP markers and the `rp-fp8` server logged 168.

| arm | mean true-decode KLD | window BCa95 |
|---|---|---|
| control-fp8 | 0.0311958 | [0.02643, 0.03776] |
| rp-fp8 | 0.0309337 | [0.02617, 0.03750] |

**rp-fp8 - control-fp8 = -0.00026 (-0.84%), paired BCa95 [-0.001146, +0.001246], lower in
19/32: no detectable change** (`results/kld-fp8-20260925/analysis.json`).

**Paired comparisons.** Each pairs the same 32 windows across two different server runs,
some on a different day, so each carries about 1-2% server-run noise. The interval is for
the per-window difference in true-decode KLD; the change is relative to the second arm's
mean.

| comparison | change | paired BCa95 | first arm lower in |
|---|---|---|---|
| FP8 KV vs NVFP4 KV, control (both 09-25) | -10.9% | [-0.00898, -0.00171] | 24/32 |
| FP8 KV vs NVFP4 KV, rp (both 09-25) | -6.4% | [-0.00337, -0.00037] | 26/32 |
| rp-fp8 vs control with NVFP4 KV (the production configuration, 09-25) | -11.7% | [-0.00892, -0.00212] | 26/32 |
| rp-fp8 vs TR3 4bpw with NVFP4 KV (09-09) | +1.5% | [-0.00315, +0.00357] | 15/32 |
| rp-fp8 vs TR3 4bpw with FP8 KV (09-09) | +9.7% | [-0.00132, +0.00640] | 13/32 |

control-fp8 vs the published 09-09 TrellisMX FP8 control (0.0319452): -2.3% in the mean. The
largest per-window difference is 0.014.

**Reading.**
- **FP8 KV is a real, significant improvement** (about -11%), matching the 09-09 measurement
  (-0.0035).
- **The row-packed remainder is not established end to end.** It shows -5.6% under NVFP4 KV
  but -0.8% under FP8 KV. Both paired intervals cross 0, so its end-to-end effect is small
  (plausibly 0-6%) and not established on 32 windows.

**FP8-KV speed.** `rp-fp8` only, with the same production config, cooling gate and single-run
caveats as Amendment 4. The NVFP4-KV runs from the earlier window are shown alongside
(`results/kld-fp8-20260925/speed/`).

| cell | control NVFP4 | rp NVFP4 | rp FP8 | rp FP8 MTP accept |
|---|---|---|---|---|
| C1 0K | 184.3 | 191.1 | 193.0 | 0.528 |
| C4 0K | 300.1 | 301.2 | 293.8 | 0.589 |
| C1 8K | 184.1 | 193.9 | 193.4 | 0.487 |
| C4 8K | 300.7 | 299.0 | 304.0 | 0.613 |

**KV capacity**, from each server's engine startup log:
- Production profile (GMU 0.88, 48 seqs): NVFP4 12,581,699 tokens; FP8 8,127,659 tokens
  (-35%).
- Capture profile (GMU 0.97, 1 seq): NVFP4 31,565,217 tokens; FP8 19,362,962 tokens.

The engine logs are not included here. The FP8 capture-profile figure is also recorded in
`results/kld-fp8-20260925/*/runtime-audit.json`.

### Amendment 6: both-hop row-pack (D-x2-RP2)

**Design.** RP2 is RP plus the same spare-row trick on the FC1 input hop of the direct decode
paths. All the new code sits behind `p8_input_rowpack`:
- the input prologue writes `x_lo = q(x - dq(x_hi))`, with its own UE8M0 byte, into token
  row +16;
- FC1 stages it into A row 8, with its scale word in SFA row 8;
- the MMA reads A rows q+8 from row 8, with the odd-lane scales from row 8, in place of the
  broadcast duplicate;
- the FC1 epilogue folds the row q+8 accumulators into row q before the FP16 store.

Paths with M > 16 are unchanged. RP2 is switched on by `B12X_P8_DOWN_REMAINDER=rp2` when the
MoE runtime is constructed. The tree is `b12x-dx2rp2` (`patches/b12x-dx2-rowpack-rp2.diff`,
`scripts/build_dx2_rp2_b12x.py`).

**Smoke test** (layer 8, rank 0):
- deterministic and finite;
- at M1/M4/M16, RP2 differs from prod by 3.7-3.9% and from RP by 2.7-2.8%;
- at M64/M512, bit-identical to prod.

**K4 closure: PASS.** Four ranks summed, Test B fit tokens (`results/k2/closure-rp2.json`;
reference values from `results/testB/`).

| layer | path | D_kernel(prod)/D_ref(P-A8) | RP2/prod (kernel) | reference both-hops P-A8x2/P-A8 |
|---|---|---|---|---|
| 3 | M1 (96 tok) | 1.0025 | 0.2531 | 0.2514 |
| 3 | M16 | 1.0015 | 0.5494 | 0.5486 |
| 8 | M1 (96 tok) | 1.0008 | 0.5154 | 0.5150 |
| 8 | M16 | 1.0003 | 0.8689 | 0.8689 |

M64 and M3072 outputs are bit-identical to prod on both layers.

**A4 timing: PASS.** RP2/prod medians over 240 blocks. They come from
`results/a4_timing_summary.json`, computed from `results/a4_timing_raw.json` with the A2/A3
estimator.

| layer | M1 | M4 | M16 |
|---|---|---|---|
| 8 K4 | 1.0263 | 1.0064 [1.0009, 1.0128] | 0.9405 |
| 3 K5 | 1.0273 | 0.9879 [0.9843, 0.9917] | 1.0024 |

The input-hop row-pack adds essentially no cost on top of the down-hop RP. On the fresh B2
layers, the layer-level both-hop damage (reference `P-A8x2`) has a geometric mean of about
0.68 (-32%), against 0.744 for down-only.

K4 and A4 both passed, so RP2 is the candidate for the final 128-window run (Amendment 7).

## Limitations

- **Development data, not qualification.** Earlier development measurements had already
  opened the 32 conditional-fit windows. This is not a measurement on untouched final
  windows, and not an independent reproduction.
- **One server run per arm.** In both windows each arm got one server preparation, in fixed
  order, so the window intervals do not capture server-run variability.
  - The direct estimates of that noise:
    - the 09-25 NVFP4-KV control vs the published 09-09 control: -1.25% in the mean and 3.2%
      per-window SD;
    - control-fp8 vs the published 09-09 FP8 control: -2.3% in the mean.
  - Every cross-run comparison carries this noise: FP8 vs NVFP4 KV across the two windows,
    against TR3, and against the 09-09 values.
- **Heavy-tailed KLD.** In the NVFP4-KV window, the top 1% of rows carry 31% of the mean.
  That window's paired interval ends at +0.000118, so a handful of rows could move the
  reading either way.
- **Layer-level proxies are not KLD.**
  - Tests B and B2 measure single-layer routed-output damage against BF16 experts on 3,072
    tokens per layer, not model output.
  - Test C used stand-in latents. Attention inputs were not captured, and with four
    hyper-connection streams the stand-in is a normalized neighbouring hidden state. A PASS
    would have needed confirmation with real latents.
- **Timing scope.**
  - Timings are isolated MoE-layer microbenchmarks on one GPU with the TP rank 0 sidecars.
  - Closure and timing cover two layers (3 and 8).
  - The zero-remainder arm is a lower bound on a real remainder's cost.
  - Test A ran on a clock-pinned GPU (see the side finding above).
- **Decode only.** D-x2-RP and D-x2-RP2 change only the M <= 16 direct paths. Prefill is
  unchanged and excluded from the score.
- **Speed.** One run per cell, descriptive only; the MTP-acceptance explanation above is open.

## In progress / pending

The status below is as of 2026-09-25. New results will be added here and in `RESULTS.md`.

### FP8-KV production window (Amendment 5): done

Results are under [Amendment 5](#amendment-5-fp8-mla-kv-window) above and in
`results/kld-fp8-20260925/`.

### FC1-input row-pack (D-x2-RP2) gates (Amendment 6): done, PASS

K4 closure and A4 timing both passed. Results are under
[Amendment 6](#amendment-6-both-hop-row-pack-d-x2-rp2) above. The RP2 builder
(`scripts/build_dx2_rp2_b12x.py`) and patch (`patches/b12x-dx2-rowpack-rp2.diff`) are
included.

### Final 128-window run (Amendment 7): running

- **Windows.** All 128 conditional-fit windows (`results/kld-cf128/verified-inputs.json`):
  - the original 32 keep their records and order;
  - the other 96 are scored for the first time;
  - the teacher logits come from the same Hugging Face dataset revision
    (`7c378d5f17dba158c4c803eff27c346dd0615660`), with every file's sha256 verified.
- **Arms.** Fixed order, one fresh server each, all with FP8 MLA KV:
  1. `control-fp8`: production P8 kernels.
  2. `rp2-fp8`: D-x2-RP2.
  3. `tr3-fp8`: a paired TR3 4bpw re-capture on the same windows.

  An `rp2-fp8` production-config speed arm follows.
- **Protocol.** The 09-09 true-decode protocol and scorer, unchanged. The capture allow-list
  is widened to the 128 window IDs.
- **Primaries.** Given in the gate table above.
- **Harness.** `scripts/window_cf128_20260925.py`.
- **Results.** Pending; they will be added next.

## Reproduction

### What you need

Public identifiers are under [Sources](#sources).
- **GPUs.** Four RTX PRO 6000 Blackwell-class GPUs for the KLD and speed runs. One GPU is
  enough for the layer-level and kernel harnesses.
- **Serving image and patched b12x.** Pull the image by digest. To get the patched b12x
  trees, either apply `patches/*.diff` to the image's b12x (`/opt/glm53-flash/b12x`) or run
  `scripts/build_*.py`.
- **TrellisMX r27 checkpoint.** Its `trellismx-manifest.json` at the cited revision is
  byte-identical to the one we ran. The sidecar sha256 values recorded in
  `results/testB/layer-003.json` and `layer-008.json` match the Hub files.
- **NVFP4 carrier.** Used by the serving image and for Test C's attention weights.
- **BF16 source.** Tests B and B2 only need the routed-expert shards of the tested layers.
- **Teacher dataset**, at two revisions:
  - the capture revision, for routed-block inputs and routes (`calibration/main-ep4-full/`,
    fit windows only; `scripts/fetch_fit_capture.py` fetches them by byte range);
  - the conditional-fit revision, for teacher logits and token arrays.
- **Campaign repository**, at the cited commit, for the `glm53_nvfp4` modules.
- **TR3 comparator (Amendment 7 only).**
  - The TR3 4bpw checkpoint at the cited revision.
  - The TR3 capture image `local/tr3-r10:cf32-process-cache-20260909` (image id
    `sha256:62e069faf47f2d42eae5f2c1677f8730a2f3f93d576301fe1bcc7e55f2fdb673`). This is a
    local build, not a published image; it is the same image that produced the published
    09-09 TR3 comparison.
  - The TR3 arm reuses that comparison's attempt-2 FP8 capture launch unchanged, apart from
    the window allow-list and the capture directory: TP4 with expert parallelism, DCP4, FP8
    KV, one sequence, 4096 batched tokens, GPU memory utilization 0.97.

Local paths in the scripts are placeholders (`<workspace>`, `<home>`, `<data-volume>`,
`<model-volume>`); see `scripts/README.md`.

### Order of runs

1. **Test A.** `scripts/run_timing.sh`, then `python3 scripts/analyze_timing.py`.
2. **Test B.** `scripts/fetch_fit_capture.py <layers>`, then `scripts/run_testB.py`.
3. **Test C.** `scripts/run_testC.py`, then `python3 scripts/analyze_BC.py`, which covers
   both B and C.
4. **K2 and K3.**
   - `scripts/dx2_base_outputs.py 3 8`
   - `scripts/run_k2.sh` (prod vs D-x2)
   - `K2_ARMS=prod K2_BUILD=image scripts/run_k2_image.sh`
   - `K2_ARMS=prod,rp K2_BUILD=rp scripts/run_k2.sh`
5. **A2, A3 and B2.**
   - `scripts/run_a2.sh`
   - `A2_ARM=rp A2_MS=1,4,16 A2_OUT=/work/results/a3_timing_raw.json scripts/run_a2.sh`
   - `scripts/run_testB2.py`
   - `python3 scripts/analyze_B2_A2.py`
6. **RP2 (Amendment 6).**
   - Build `b12x-dx2rp2` with `DX2_DST=<path> scripts/build_dx2_rp2_b12x.py`, or apply
     `patches/b12x-dx2-rowpack-rp2.diff`.
   - Smoke: `scripts/run_rp2_smoke.sh`.
   - K4: `B12X_TREE=b12x-dx2rp2 K2_ARMS=prod,rp2 K2_BUILD=rp2 scripts/run_k2.sh`.
   - A4: `B12X_TREE=b12x-dx2rp2 A2_ARM=rp2 A2_MS=1,4,16 A2_OUT=/work/results/a4_timing_raw.json scripts/run_a2.sh`,
     then `python3 scripts/summarize_rowpack_timing.py results/a4_timing_raw.json results/a4_timing_summary.json "RP2/prod"`.
7. **End-to-end KLD.**
   - `scripts/window_nvfp4_20260925.py` (Amendment 4), `scripts/window_fp8_20260925.py`
     (Amendment 5) and `scripts/window_cf128_20260925.py` (Amendment 7).
   - The inputs for the 128-window run come from `scripts/build_inputs_128.py`.
   - The window scripts need the 09-09 reference launch arguments and a capture-only
     derivative of the serving image. The 128-window run also needs the TR3 capture launch
     and image.
   - They also operate our production host (systemd user unit, locks, ports). Treat them as
     a record of the procedure and adapt them before use.

### Re-deriving the reported numbers (no GPU)

- **Timing and layer-level analyses.** `analyze_timing.py`, `analyze_timing_split.py`,
  `analyze_BC.py` and `analyze_B2_A2.py` regenerate `timing_analysis.json`,
  `timing_split_analysis.json`, `analysis_BC.json` and `analysis_amendment2.json` exactly
  from the raw files in `results/`.
- **A3 and A4.** `summarize_rowpack_timing.py` gives the medians and intervals reported for
  A3 and A4 from `a3_timing_raw.json` and `a4_timing_raw.json`.
- **KLD.** Both windows (`results/kld-rp-20260925/`, `results/kld-fp8-20260925/`) include the
  per-window score records (`scores/*.json`) and per-row scores (`scores/*.npz`). The `.npz`
  files hold row KL, teacher entropy, top-1 tokens and probabilities, and realized-token
  log-probabilities.
  - The intervals in each `analysis.json` are `scipy.stats.bootstrap` BCa over the
    per-window `true_decode_mean_kld` values.
  - Each `true_decode_mean_kld` is the mean of `kld[1:]` in the matching `.npz`.
  - The cross-run comparisons use the same estimator on window-paired differences between
    these files and the published 09-09 `comparison.json` files.

**Not included:** `.pt` tensors (K2 inputs and outputs, Test C mirror tensors), raw logits
(retired after hashing and scoring), request/response captures, server and startup logs,
telemetry, and clock CSVs.

## Sources

### Models, data, code and images we used

1. **Base model.** `zai-org/GLM-5.3-Flash-BF16`, revision
   `a6c167b62691b2bac901344b65cb651a70f53e43`
   (https://huggingface.co/zai-org/GLM-5.3-Flash-BF16). Used for the teacher logits and as the
   BF16 source experts in Tests B and B2.
2. **NVFP4 carrier.** `local-inference-lab/GLM-5.3-Flash-NVFP4`, revision
   `520de24eabf507659eaef7c70f14fd584527facc`
   (https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4). Supplies the non-routed
   weights of the TrellisMX runtime (the checkpoint manifest pins this revision) and Test C's
   attention weights.
3. **TrellisMX r27 checkpoint.** `brandonmusic/GLM-5.3-Flash-TrellisMX-MXFP8`, revision
   `b492969185c600f2dc431fe12acbbdf2632e905b`
   (https://huggingface.co/brandonmusic/GLM-5.3-Flash-TrellisMX-MXFP8). This is the checkpoint
   under test. The same revision holds the published 09-09 KLD results we compare against:
   `results/kld-reference-20260909/` (TrellisMX, FP8 and NVFP4 MLA KV) and
   `results/kld-tr3-20260909/` (matched TR3 comparison).
4. **TR3 comparator.** `brandonmusic/GLM-5.3-Flash-tr3-4bpw`, revision
   `aba59d2175e1ee2887ae0ae1300ba848b1deed84`
   (https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw).
   - Its 09-09 KLD values come from the published matched-data comparison in item 3.
   - The Amendment 7 re-capture uses the same model revision and the same capture image,
     `local/tr3-r10:cf32-process-cache-20260909` (image id
     `sha256:62e069faf47f2d42eae5f2c1677f8730a2f3f93d576301fe1bcc7e55f2fdb673`). That image
     is a local build and is not published.
5. **Teacher logits and captures.** Dataset `brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits`
   (https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits):
   - revision `95f4fdd94bf29989db2e0d1054e4931f55edb6aa`: routed-block captures;
   - revision `7c378d5f17dba158c4c803eff27c346dd0615660`: conditional-fit teacher logits and
     token arrays.
6. **Campaign repository.** https://github.com/brandonmmusic-max/glm53-hadamard-shapleymcg-kld,
   commit `03ff56b20a5934748ca5117010c24aad9c208536`. Reused modules in `glm53_nvfp4`:
   - `p8_layer_rate_damage`, `p8_coupled_scale`, `capture`, `shard_index`;
   - `canary_mxfp6_reap` (`_qdq_e4m3_k32`, `_metrics`);
   - `screen_p8_h128_activation` (`_ordinary`);
   - `p8_decode_protocol`, the KLD capture/scoring protocol. It was imported from a local
     checkout of the same repository and is byte-identical to the file at this commit.

   The published 09-04 screen value reproduced in Test B is in
   `results/P8_COUPLED_INCOHERENCE_ARCHIVE_AUDIT.md` of that repository.
7. **Serving image.** `verdictai/trellismx:glm53-flash-p8-r27-reference-20260909`, digest
   `sha256:ca6b80188dce154b91f49108b7d87792d2ba6328935afc71b44d1c0e6f6a1adf`.
   - Its published image record names the base image
     `voipmonitor/vllm:jovian-judgement-community-20260906-r27@sha256:a298fe1cd207eaf97bd2ff2686716ed25b7009c09b36650eba732a4a7dc51512`.
   - The vLLM fork inside it reports
     `0.26.1rc0+glm53.flash.nvfp4.luke.clean.r1.vllme75bcfd.b12x58a046f`.
8. **b12x.** Luke Alonso and contributors, https://github.com/local-inference-lab/b12x (the
   earlier https://github.com/lukealonso/b12x redirects there). Apache License 2.0.
   - We patched the copy of b12x inside the serving image. Its git origin is
     local-inference-lab/b12x.
   - The 16 existing files that the patches modify are identical to commit
     `d564f6ca54c092497ec5ae7e07a272f55eec7dbe` of the TrellisMX integration branch
     (https://github.com/brandonmmusic-max/b12x). The patches also add one new file,
     `p8_down_remainder.py`.
9. **Decode benchmark.** `llm_decode_bench.py` version 0.4.29 from
   https://github.com/local-inference-lab/llm-inference-bench (the script's own update
   URL). sha256 of the copy we ran:
   `7239958032ec10781d7db3efca23a7602bd9aeb94455a723e2634ab7d6fe546a`.

### Background

- **QTIP.** Tseng, A., Sun, Q., Hou, D., and De Sa, C. "QTIP: Quantization with Trellises and
  Incoherence Processing." NeurIPS 2024. arXiv:2406.11235. The trellis-coded quantization
  with compute-based codes that EXL3 builds on.
- **ExLlamaV3 / EXL3.** turboderp. ExLlamaV3 and the EXL3 format.
  https://github.com/turboderp-org/exllamav3. The b12x source headers state that the P8
  decode's procedural MCG constants and state construction are ported from ExLlamaV3's
  procedural MCG decoder, and the b12x intrinsics label the trellis path "QTIP/EXL3".
- **Microscaling (MX).** Open Compute Project, "OCP Microscaling Formats (MX) Specification
  v1.0," 2023. Rouhani, B. D., et al. "Microscaling Data Formats for Deep Learning."
  arXiv:2310.10537, 2023. These define E4M3 elements with UE8M0 scales per 32-element block,
  as used by the P8 activations.
- **NVFP4 and block-scaled MMA.** NVIDIA: NVFP4 (E2M1 values with E4M3 scales per 16
  elements and a second-level scale), and the PTX ISA block-scaled `mma.sync`
  (`kind::mxf8f6f4`). https://docs.nvidia.com/cuda/parallel-thread-execution/
- **QuaRot.** Ashkboos, S., Mohtashami, A., Croci, M., Li, B., Jaggi, M., Alistarh, D.,
  Hoefler, T., and Hensman, J. "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs."
  NeurIPS 2024. arXiv:2404.00456. Rotation before low-bit quantization, the idea tested in
  Test C.
- **Kimi Linear (KDA).** Moonshot AI. "Kimi Linear: An Expressive, Efficient Attention
  Architecture." arXiv:2510.26692, 2025.
- **DeepSeek Sparse Attention.** DeepSeek-AI, DeepSeek-V3.2-Exp, 2025.
  https://github.com/deepseek-ai/DeepSeek-V3.2-Exp. With Kimi Linear, the two attention
  types in GLM-5.3-Flash's hybrid layout (Amendment 1).
- **KL divergence.** Kullback, S., and Leibler, R. A. "On Information and Sufficiency."
  Annals of Mathematical Statistics 22(1), 1951.
- **BCa intervals.** Efron, B. "Better Bootstrap Confidence Intervals." Journal of the
  American Statistical Association 82(397), 1987.
- **SciPy.** `scipy.stats.bootstrap` (`method="BCa"`), used for the KLD intervals.
- **Preregistration.** Nosek, B. A., Ebersole, C. R., DeHaven, A. C., and Mellor, D. T. "The
  preregistration revolution." PNAS 115(11), 2018.

## License

This repository's own scripts, documentation and result files are distributed under the
ShapleyMCG License 1.0 (`LICENSE`), the same license as the campaign repository. It is
source-available, not OSI open source. Required attribution:

> ShapleyMCG was created by Brandon M. Music. Canonical source:
> https://github.com/brandonmmusic-max/shapleymcg

The b12x kernel patches in `patches/` modify Apache-2.0 b12x sources and are provided under
the Apache License 2.0 (`patches/LICENSE.b12x`). Third-party models, datasets, images and
libraries keep their own licenses.

## Repository layout

```
README.md                 this write-up
PREREG.md                 preregistration and Amendments 1-7
RESULTS.md                results log, written as results came in
LICENSE                   ShapleyMCG License 1.0
scripts/                  every script that produced a number (see scripts/README.md)
patches/                  b12x kernel diffs and their Apache-2.0 license (see patches/README.md)
results/
  timing_raw.json, timing_analysis.json                Test A
  timing_split_raw.json, timing_split_analysis.json    Test A exploratory split
  gpucheck_*.json                                      per-GPU M1 timing and clock checks
  testB/                                               Test B per-layer records, 09-04 screen reproduction
  testB2/                                              B2 per-layer records
  testC/                                               Test C per-configuration records
  analysis_BC.json                                     Tests B and C analysis
  k2/closure-*.json                                    K2, K3 and K4 closure
  a2_timing_raw.json, a3_timing_raw.json               A2 and A3 timing
  a4_timing_raw.json, a4_timing_summary.json           A4 timing (raw and summary)
  dx2_split_timing_raw.json                            A2 exploratory split
  analysis_amendment2.json                             B2 and A2 analysis
  kld-rp-20260925/                                     Amendment 4 production window:
    analysis.json                                        KLD analysis
    verified-inputs.json                                 window list, teacher revision, input hashes
    control/, rp/                                        runtime audits, per-window scores (.json, .npz)
    speed/                                               benchmark outputs and launch arguments
    window.log, execution-final.json, restoration.json   window log and production restoration record
  kld-fp8-20260925/                                    Amendment 5 production window (FP8 MLA KV):
    analysis.json, verified-inputs.json                  KLD analysis, window list and input hashes
    control-fp8/, rp-fp8/                                runtime audits (with KV capacity), per-window scores
    speed/rp-fp8/                                        benchmark outputs and launch arguments
    window.log, execution-final.json, restoration.json   window log and production restoration record
  kld-cf128/verified-inputs.json                       Amendment 7 inputs: all 128 windows, hashes (run in progress)
```
