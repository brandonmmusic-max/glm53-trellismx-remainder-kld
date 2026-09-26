# FP8 remainder carriers and NVFP4-KV rotation on TrellisMX P8 kernels (GLM-5.3-Flash)

Brandon M. Music, September 2026

**Status: complete (2026-09-26).**

This repository is the full record of one preregistered study:
- the preregistration and its seven amendments (`PREREG.md`);
- the results log, written as results came in (`RESULTS.md`);
- every script that produced a number (`scripts/`);
- the raw and analysed result files (`results/`);
- the kernel patches (`patches/`).

This README explains what was tested, why, how, and what came out, including every test that
failed.

## Summary

We serve GLM-5.3-Flash from our TrellisMX r27 checkpoint on custom mixture-of-experts kernels
called "P8". The routed-expert weights are trellis-coded, and the kernel decodes them to 8-bit
floating point (E4M3) inside the matrix multiply. Activations are rounded to E4M3, with one
power-of-two scale per 32 values.

The study asked two questions:
- Can a second FP8 "remainder" term for the activations bring the model measurably closer to
  its BF16 original without slowing decode?
- Can a fixed rotation make the 4-bit (NVFP4) attention KV cache more accurate?

The findings, in the order the tests ran:

1. **A second FP8 multiply per decoded weight fragment is not free at decode.** It cost 7-17%
   of MoE-layer time. (Test A: FAIL.)
2. **At layer level, two-term activations remove about half the error.** Carrying activations
   as two FP8 terms (hi + lo) removes 51.0% of the routed-output error against the BF16
   experts. (Test B.)
3. **A fixed 512-point Hadamard rotation before the NVFP4 MLA KV cache does nothing.** The cached
   latent is already Gaussian-shaped. (Test C: FAIL.)
4. **A second data plane for the remainder is correct but too slow.** Adding the remainder only
   at the down-projection input, in a second data plane, computes the right numbers (closure
   PASS) but costs up to 19% of layer time (timing FAIL). (Amendment 2.)
5. **Row-packing makes the remainder nearly free on the decode kernels.** The remainder goes into
   a row of the tensor-core tile that those kernels already compute and throw away. This was
   done first at the down-projection input (RP), then at both inputs (RP2), and every gate
   passed. (Amendments 3 and 6.)
6. **The final, preregistered end-to-end run** used all 128 conditional-fit windows with an FP8
   KV cache (Amendment 7).
   - The both-hop row-pack RP2 lowers true-decode KL divergence from the BF16 teacher by
     **4.9%** (95% CI -7.3% to -1.8%) relative to the production kernels. Under the
     preregistered rule that is an improvement.
   - It does **not** reach TR3 4bpw: RP2 stays **8.3%** above it (95% CI +2.0% to +14.5%) and
     is not equivalent within +/-5% or +/-10%.
   - RP2 closes about 40% of the production kernels' gap to TR3.
7. **The KV cache mattered more.** Switching the attention KV cache from NVFP4 to FP8 lowered KL
   divergence by 10.9% (significant), at a cost of 35% of KV capacity. The earlier 32-window
   checks of the down-hop-only RP were inconclusive: -5.6% with NVFP4 KV and -0.8% with FP8 KV,
   and both intervals crossed 0.

**Scope.** RP and RP2 change only the decode kernels that handle up to 16 tokens per step. The
KL measurement pushes every token through those kernels. In serving, prompts are prefilled by
unchanged kernels, so the served effect is expected to be smaller. That was not measured.

| Question | Test | Outcome |
|---|---|---|
| Is a second FP8 MMA per decoded weight fragment free at decode? | Test A: zero-remainder timing | **FAIL.** Median time ratio 1.0698 / 1.0854 at M1 and 1.1650 / 1.1484 at M4 (layers 8 / 3); the bar was <= 1.03. |
| How much routed-output damage does a two-term (hi + lo) FP8 activation carrier remove? | Test B: layer level, 6 layers | Pooled damage ratio 0.490 (-51.0%), BCa 95% CI [0.427, 0.562]; 6/6 layers improve. |
| Does a fixed H512 rotation before the NVFP4 MLA latent record reduce error? | Test C: 6 MLA layers | **FAIL.** Pooled ratio 1.0011 for latent NMSE and 1.0055 for attention-logit error. |
| Down-hop remainder in a second data plane (D-x2) | K2 / A2 / B2 | Closure PASS. Timing FAIL: 1.071-1.079 at M4, 1.163-1.187 in prefill. Fresh-layer damage 0.744 (-25.6%), B2 PASS. |
| Down-hop remainder packed into the idle MMA row +8 (D-x2-RP) | K3 / A3 | Closure PASS. Timing PASS: median ratio <= 1.0272 at M1 and M4. |
| End-to-end KLD, NVFP4 MLA KV, 32 windows | Amendment 4 | control 0.0350129 vs rp 0.0330631: -5.57%, paired BCa 95% [-0.005975, +0.000118]: **no detectable change**. |
| End-to-end KLD, FP8 MLA KV, 32 windows | Amendment 5 | control-fp8 0.0311958 vs rp-fp8 0.0309337: -0.84%, [-0.001146, +0.001246]: **no detectable change**. FP8 vs NVFP4 KV for the control: -10.9% [-0.00898, -0.00171]. |
| Remainder packed at both hops (D-x2-RP2) | K4 / A4 | Closure PASS. Timing PASS: median ratio <= 1.0273 at M1 and M4. |
| **Final: 128 windows, FP8 MLA KV** | Amendment 7 | rp2 - control **-4.9% [-7.3%, -1.8%]**, lower in 105/128: **improvement**. rp2 - TR3 **+8.3% [+2.0%, +14.5%]**: **not tied**, not equivalent at +/-5% or +/-10%. |

## 1. What was tested, and why

The published 09-09 comparison measured our TrellisMX checkpoint against TR3 4bpw, the uniform
4-bit trellis checkpoint of the same model, on the same windows. TrellisMX had higher KL
divergence from the BF16 teacher:

| KV cache | TrellisMX | TR3 4bpw |
|---|---|---|
| FP8 | 0.0319452 | 0.0281899 |
| NVFP4 | 0.0354562 | 0.03048 |

This study looked for inexpensive ways to close that gap on the TrellisMX side without touching
the weights. There were two ideas.

**An FP8 remainder for the activations.**
- A P8 layer has two error sources: the trellis-coded weights, and the E4M3 rounding of the
  activations that enter each of its two matrix multiplies.
- In Test B, swapping the P8 weights for the exact BF16 weights, with production activations
  kept, still left 13-59% of the routed-output error on the tested layers (the "activation
  share" column below). Carrying the activations as two FP8 terms, a value plus its rounded
  remainder, attacks that share directly.
- The open question was cost. On a trellis kernel, decoding the weights is the expensive step,
  so reusing a decoded weight fragment for a second multiply might be close to free.
- Test A measured this, and it was not free. The row-pack variants (RP and RP2) then found a
  place in the decode kernels where the second multiply really is free.

**A rotation for the NVFP4 KV cache.** Production stores the attention cache as a 4-bit NVFP4
record. A fixed orthogonal rotation before quantization spreads outlier channels across the
vector, as in QuaRot. Its inverse could be folded into the attention weights, so it would cost
nothing at runtime. Test C checked whether there are outliers to spread. There are not.

## 2. Background

### 2.1 The P8 kernels

In the TrellisMX r27 checkpoint, the routed experts of GLM-5.3-Flash's 42 MoE layers are trellis
bitstreams at 4 or 5 bits per weight ("K4" or "K5"), chosen per layer. All other weights come
from an NVFP4 carrier checkpoint.

The P8 kernels live in b12x:
- **Weights.** Each weight fragment is decoded in registers to E4M3 with a procedural MCG
  codebook. Its constants and state construction are ported from ExLlamaV3's procedural MCG
  decoder, and that family of trellis codes descends from QTIP. The decoded fragment goes
  straight into a block-scaled tensor-core matrix multiply ("MMA": `mma.sync`,
  `kind::mxf8f6f4`, shape m16n8k32).
- **Activations.** E4M3 values with a power-of-two UE8M0 scale per 32 elements along the
  reduction dimension ("E4M3/UE8M0-K32", the OCP Microscaling layout). They are quantized at two
  points ("hops"):
  - the FC1 (gate/up projection) input;
  - the FC2 (down projection) input after SwiGLU.
- **Coupled transform.** The checkpoint's block Hadamard transforms and per-channel scales sit
  around both multiplies.
- **Kernel paths.** "M" below is the number of tokens in one MoE-layer call.
  - M <= 16 (decode steps) uses "direct" small-M kernels.
  - M 17-128 uses grouped 32-row tiles.
  - Larger M uses the prefill kernels.

### 2.2 The remainder (two-term) carrier

E4M3 keeps 3 mantissa bits. The two-term carrier sends two terms, each with its own UE8M0 scale:
- `hi = q(x)`;
- `lo = q(x - dq(hi))`, what is left after subtracting `hi`.

In a kernel this is a second MMA on the same decoded weight fragment, accumulating into the same
FP32 accumulator, because `x W ~= hi W + lo W`.

### 2.3 Row-packing on one-route decode tiles

On the direct decode paths (M <= 16), every physical 16/32-row MMA tile holds exactly one route
row (`valid_rows == 1`), where a route row is one token sent to one expert. The other rows of the
A operand are zeros that the tensor core multiplies anyway. In the m16n8 accumulator layout, each
thread holds the same output columns for rows q and q+8.

**D-x2-RP** (the down-projection remainder) uses that spare row:
- **FC1 epilogue.** Writes `lo = q(v - dq(hi))`, with its own UE8M0 byte, into row +8 of the
  route's own tile.
- **FC2.** Its main loop is unchanged: it already stages and multiplies row 8, and its odd lanes
  already supply the row-8 scale.
- **FC2 epilogue.** Folds row 8 into row 0 in registers before the FP16 store
  (`fragment[0] += fragment[2]`, `fragment[1] += fragment[3]`).

**D-x2-RP2** adds the same trick at the FC1 input:
- the input prologue writes `x_lo = q(x - dq(x_hi))` into token row +16;
- FC1 stages it into A row 8, in place of the broadcast duplicate that row used to hold;
- the FC1 epilogue folds the row q+8 accumulators into row q.

Neither variant adds an MMA, a staging pass or a second data plane. Paths with M > 16 keep the
production single-term kernels, so both variants affect decode only.

### 2.4 The NVFP4 MLA KV record

Production stores the MLA latent cache (512 dimensions, no RoPE payload) as a two-level NVFP4
record:
- a per-token scale `s_t = amax_t / (6 * 448)`;
- E4M3 scale bytes per 16 elements, relative to `s_t`;
- E2M1 (4-bit) values.

A rotation helps such a format only when a few channels dominate the per-group maxima.

## 3. How it was tested, and why each choice

### 3.1 Preregistration, with every amendment written before its data

**What.** `PREREG.md` was written on 2026-09-23, before any number existed. It fixes each test's
harness, metric, uncertainty estimator and decision rule. Seven amendments followed. Each was
written before any number it governs, and each says what was already known at the time.

**Why.** Almost every choice here could be tuned after seeing results: the layers, the windows,
the thresholds, the estimator. Fixing them in advance removes the option of picking the analysis
that looks best.
- Amendments let the plan change for good reasons without hiding the change. For example,
  Amendment 1 moved Test C to the layers that actually carry an MLA cache.
- Exploratory analyses are labelled as such and cannot change a decision.
- A preregistered FAIL stops its direction. Test A's FAIL ended the two-hop design whatever
  Test B showed, and the down-hop work restarted under Amendment 2 on fresh data.

| Part | Written | Content |
|---|---|---|
| Original | 2026-09-23 | Tests A (timing), B (layer numerics) and C (KV rotation), with decision rules. |
| Amendment 1 | 2026-09-23, before any Test C number | Moves Test C to the model's actual MLA layers: only every fourth layer has the MLA cache, and the rest use KDA linear attention. Records that Test A had already failed. |
| Amendment 2 | 2026-09-23, before any of its numbers | Confirms the post-hoc down-hop signal on fresh layers (B2) and on the real kernel: closure (K2) and timing (A2). |
| Amendment 3 | 2026-09-23, before any D-x2-RP number | Row-packed down-hop remainder: closure (K3) and timing (A3). |
| Amendment 4 | 2026-09-25, before any end-to-end number | Decode-scored KLD with the 09-09 reference protocol and descriptive speed, in an approved production window (NVFP4 KV). |
| Amendment 5 | 2026-09-25, before any number | Second window with FP8 KV, then the FC1-input row-pack. |
| Amendment 6 | 2026-09-25, before any D-x2-RP2 number | Gates for the both-hop row-pack RP2: closure (K4) and timing (A4). |
| Amendment 7 | 2026-09-25, before any 128-window number | Final run on all 128 conditional-fit windows with FP8 KV: arms `control-fp8`, `rp2-fp8` and `tr3-fp8`, two preregistered primaries, and an `rp2-fp8` speed arm. |

### 3.2 Layer-level gates before production windows

**What.** Every design had to pass offline gates before any end-to-end run:
- layer numerics against the BF16 experts (Tests B and B2);
- real-kernel closure (K2-K4);
- real-kernel timing (Test A, A2-A4).

**Why.** An end-to-end KL run needs the production server stopped; the final three-arm capture
alone took about three hours. The offline gates run on one GPU while production keeps serving.
They stopped two designs, the two-hop remainder and the plane-based D-x2, before any production
time was spent on them.

| Gate | PASS requires |
|---|---|
| Test A | Median paired time ratio <= 1.03 at M1 and M4 on both layers, and every output exact. |
| Test B | Pooled geometric-mean damage reduction >= 5%, >= 5 of 6 layers improve, and the pooled log-ratio CI excludes 0. |
| Test C | Both metrics drop >= 20% pooled, and the CI excludes 0. |
| B2 | Pooled down-only reduction >= 20%, >= 5 of 6 layers improve, and the upper CI bound of the pooled log ratio < 0. |
| K2 / K3 / K4 | Per layer and path: `abs(D_kernel(prod) / D_ref(P-A8) - 1) <= 5%`, and the arm/prod ratio within +/-0.05 of the reference ratio (down-only for K2/K3, both hops for K4). K3 and K4 also require M64/M3072 outputs bit-identical to prod. |
| A2 | Decode: median <= 1.03 at M1 and M4. Prefill: median <= 1.05 at M512 and M2048. |
| A3 / A4 | Median <= 1.03 at M1 and M4 on both layers. |
| Amendments 4 and 5 reading | Improvement if the paired mean < 0 and the CI excludes 0. No detectable change if the CI includes 0. Harm if the mean > 0 and the CI excludes 0. |
| Amendment 7 primaries | (1) `rp2-fp8 - control-fp8`: improvement if the mean < 0 and the CI excludes 0. (2) `rp2-fp8 - tr3-fp8`, reported three ways: whether the CI includes 0, whether the whole CI lies within +/-5% of TR3's mean, and the same at +/-10%. |

### 3.3 Layer-level numerics (Tests B and B2)

**What.** `scripts/remainder_damage.py` uses the campaign's `p8_layer_rate_damage` method:
- the r27 rank sidecars are reference-decoded with the kernel's procedural-MCG E4M3 codebook;
- the exact coupled reference forward runs over all 288 experts;
- the routed-output damage is `D = sum_t ||S(t)||^2 / sum_t ||routed_output(t)||^2`, against
  the BF16 source experts, on 64 fit windows x 48 tokens = 3,072 tokens per layer.

The activation carrier is selectable at each hop:

| Arm | Weights | FC1 input | Down input |
|---|---|---|---|
| `P-A8` (production) | P8 | single-term | single-term |
| `P-A8x2` | P8 | two-term | two-term |
| `P-Aexact` (weight-only error) | P8 | exact | exact |
| `B-A8`, `B-A8x2` (activation-only error) | BF16 in coupled coordinates | single / two-term | single / two-term |
| `B-Aexact` (closure) | BF16 in coupled coordinates | exact | exact |
| `P-A8x2-fc1only`, `P-A8x2-downonly` (added after Test A, reported only) | P8 | two-term / single-term | single-term / two-term |

**Why.** Routed-output damage is cheap to compute on one GPU and isolates exactly the part of the
model that the remainder changes. The weight-only and activation-only arms show how much of the
error is reachable at all.

**Checks.**
- The harness asserts that the production arm is bit-identical to the campaign's
  `coupled_expert_reference`.
- The BF16 closure arm lands at 4-9e-8 relative.
- A like-for-like rerun of an earlier (09-04) activation screen reproduces its published value
  to a relative difference of 1e-7, so the harness measures the same thing the earlier work did.

**Uncertainty.** A paired window BCa bootstrap, where one resample of the 64 window indices is
applied to every layer at once (20,000 replicates, seed 20260923, jackknife acceleration;
`scripts/analyze_BC.py`). The statistic is the mean over layers of
`log(sum_w D_new / sum_w D_ref)`; its exponential is the pooled geometric-mean ratio.

### 3.4 KV rotation (Test C)

**What.**
- **Stand-in latents.** Attention inputs are not in the capture, so the latent for MLA layer A
  is built from the routed-block capture of layer C:
  - `a = gamma_in(A) * (m_C / gamma_post(C))`;
  - `c = kv_a_layernorm(kv_a_proj(a))`, using the carrier's BF16 attention weights.
- **Queries.** `q_a_proj -> q_a_layernorm -> q_b_proj`, absorbed with each head's `W_UK`.
- **Layers.** The decisive pairs A:C are 7:6, 15:14, 19:18, 27:26, 35:34 and 43:42.
- **Quantizer.** A torch mirror of b12x's two-level record writer, checked against the real
  `concat_and_cache_glm_next_mla`.
- **Arms.** Unrotated; normalized Sylvester H512 (decisive); H512 with fixed random signs
  (reported only).
- **Metrics.** Latent NMSE, and relative attention-logit error
  `sum (q_abs . (c_hat - c))^2 / sum (q_abs . c)^2`.

**Why.** If a rotation cannot reduce the error of the cached latent, it cannot help the attention
that reads it, and there is no reason to spend a production window on it.

### 3.5 Closure tests (K2, K3, K4)

**What.** `scripts/dx2_closure.py` runs on layers 3 (K5) and 8 (K4).
- All four tensor-parallel rank sidecars run one at a time, and their outputs are summed, which
  is what the all-reduce computes.
- The Test B fit tokens are fed in chunks that exercise every kernel path: M1, M16, M64 and
  M3072.
- Damage is measured against the BF16-source routed output.
- K4 uses the RP2 tree (`B12X_TREE=b12x-dx2rp2`) and the `rp2` arm.

**Why.**
- The layer-level numbers come from a reference implementation. Closure proves that the CUDA
  kernels compute the same thing, so the reference numbers transfer to the kernels.
- Closure also catches layout and scale bugs that a speed test cannot see.
- The bit-identity checks on untouched paths prove that the new code does not leak into the
  production kernels.

### 3.6 Timing with CUDA graphs, interleaved blocks and paired ratios

**What.** `scripts/timing_harness.py` and `scripts/dx2_timing.py`.
- `P8NativeTPMoE` is built for tensor-parallel rank 0 the way the serving integration builds
  it, with real routed-block inputs and top-8 routes.
- Each timed unit is a CUDA-graph replay of one MoE-layer call, 200 replays per block.
- The two arms alternate block by block: 60 blocks per cell. Decode cells use 4 input sets
  (240 blocks) and prefill cells use 2.
- The metric is the median of per-block paired time ratios, with a percentile-bootstrap interval
  of that median (20,000 resamples, seed 20260923).
- Compile caches were off.

**Why.**
- Graph replay removes Python and launch overhead, so only kernel time is measured.
- Alternating the arms puts both under the same clock and temperature conditions, so drift
  cancels in the ratio.
- The median resists outlier blocks.
- Compile caches key on the compile spec rather than on the arm, and could otherwise mix arms.

### 3.7 True decode with forced tokens

**What.** The end-to-end protocol is the unchanged 09-09 reference protocol.
- The capture server runs with multi-token prediction (MTP) off and one sequence.
- Each 2,048-token window is decoded step by step, with each next token forced to the window's
  text.
- The server records the full logits at every step: 2,047 predictions.
- Row 0 comes from prefill and is excluded, leaving 2,046 true-decode rows per window.

**Why.**
- RP and RP2 change only the decode kernels. A prefill-scored KL measurement would push every
  token through the unchanged prefill kernels and could not see them.
- Forcing the tokens keeps every arm on an identical history, so rows line up exactly across
  arms and the comparison is row for row.
- Reusing the published protocol makes our controls comparable with the published 09-09
  values.

### 3.8 KL divergence in FP64 over the full vocabulary

**What.** For each row, `KL(teacher || student)` over all 154,880 tokens, computed on CPU in FP64
from the stored FP32 teacher logits of the BF16 model.

**Why.**
- KL divergence from the BF16 teacher measures how far the whole next-token distribution moves,
  not just the top choice.
- The full vocabulary avoids the bias of top-k truncation.
- FP64 keeps summation error far below effects of a few percent.

### 3.9 Windows as the resampling unit, with paired BCa intervals

**What.** Each window's mean KL over its 2,046 rows is one observation, and arms are compared
through per-window differences. Intervals come from `scipy.stats.bootstrap` with
`method="BCa"`, 20,000 resamples and `random_state=20260902`, the estimator of the published
intervals.

**Why.**
- Rows within a window share context and are strongly correlated, so treating 2,046 rows as
  independent would overstate precision. The window is the independent unit.
- Window means span more than an order of magnitude (`results/kld-cf128/analysis.json`), and
  pairing removes that difficulty spread from the comparison.
- KL is heavy-tailed: in the first production window, the top 1% of rows carried 31% of the
  mean. BCa corrects the bias and skew of the bootstrap distribution better than a plain
  percentile interval.
- The fixed seed makes every interval reproducible.

### 3.10 Only conditional-fit windows

**What.** The teacher dataset assigns its windows to roles: fit, conditional-fit, selection,
confirmation and final.
- This study read only fit windows (layer tests) and conditional-fit windows (KL runs).
- The first two production windows reused 32 conditional-fit windows that the 09-09
  measurements had already opened.
- The final run used all 128 conditional-fit windows, 96 of them scored for the first time.

**Why.**
- Selection, confirmation and final windows are held back for later qualification. Using them
  here would spend held-out data on development.
- Reusing the 32 opened windows kept the first runs comparable with the published values.
- Adding the 96 unopened conditional-fit windows keeps the final result from resting only on
  windows that had been looked at before.

### 3.11 FP8 KV cache in the final run

**What.** All three arms of the final run used FP8 MLA KV.

**Why.**
- The FP8-KV window showed that FP8 KV lowers KL by 10.9% for our control.
- The published 09-09 values show the same direction for TR3: 0.0281899 with FP8 KV, against
  0.03048 with NVFP4 KV.
- The final comparison is therefore made at each system's better cache.

### 3.12 Fixed arm order, with run-to-run drift measured

**What.** One production window and one fresh server per arm, in a fixed order: control, then
RP2, then TR3.

**Why.**
- Fresh servers rule out state carried from one arm to the next.
- A fixed order could confound slow drift with the arm, so the drift was measured on the
  original 32 windows:
  - the control moved -0.07% against the same day's FP8 window;
  - TR3 moved -1.07% against its 09-09 run.
- About 1% or less is well below the 8.3% RP2-TR3 gap.

### 3.13 Estimators and seeds

| Analysis | Estimator | Resamples | Seed |
|---|---|---|---|
| Test A, exploratory split, A2, A3 and A4 timing | percentile bootstrap of the median over blocks | 20,000 | 20260923 |
| Tests B, C and B2 | paired window BCa with jackknife acceleration | 20,000 | 20260923 |
| Test C random-sign arm | fixed random signs | n/a | 20260923 |
| End-to-end KLD, including cross-run comparisons | `scipy.stats.bootstrap`, BCa, over window-paired differences | 20,000 | 20260902 |

### 3.14 Model, checkpoint, runtime and hardware

- **Base model.** `zai-org/GLM-5.3-Flash-BF16`. It is the KL teacher and the reference for the
  layer-level damage measurements.
- **Checkpoint.** TrellisMX r27 (`brandonmusic/GLM-5.3-Flash-TrellisMX-MXFP8`): per-rank trellis
  sidecars for the routed experts, four tensor-parallel ranks per layer, overlaid on the NVFP4
  carrier `local-inference-lab/GLM-5.3-Flash-NVFP4`.
- **Runtime.** Serving image `verdictai/trellismx@sha256:ca6b8018...` (full digest under
  [Reproduction](#6-reproduction)). It is vLLM-based, and the server reports
  `0.26.1rc0+glm53.flash.nvfp4.luke.clean.r1.vllme75bcfd.b12x58a046f`. Every patched arm
  bind-mounts a patched b12x tree over `/opt/glm53-flash/b12x`.
- **Serving layout.**
  - Capture profile: TP4/DCP4 (tensor parallel 4, decode context parallel 4), MTP off, one
    sequence, 4096 batched tokens, GPU memory utilization 0.97, maximum length 1M.
  - Production speed profile: MTP with 3 speculative tokens, 48 sequences, 8192 batched tokens,
    GPU memory utilization 0.88.
- **Hardware.** Four NVIDIA RTX PRO 6000 Blackwell GPUs (two Max-Q, two Workstation Edition),
  PCIe, 300 W limit each.

## 4. Results

Every number below comes from `RESULTS.md` or the files under `results/`, copied at the
precision of the source.

### 4.1 Test A: zero-remainder timing (preregistered). FAIL

**Setup and checks.**
- 60 interleaved blocks x 200 replays x 4 input sets per cell, on GPU 0.
- Every output was bit-identical to the baseline, and the baseline was deterministic.
- The unmasked positive control changed the outputs.

Median time ratio (arm Z / baseline) with 95% CI (`results/timing_analysis.json`):

| layer | M1 | M4 | M16 (reported only) |
|---|---|---|---|
| 8 (K4) | 1.0698 [1.0697, 1.0699] | 1.1650 [1.1648, 1.1653] | 1.1341 |
| 3 (K5) | 1.0854 [1.0853, 1.0855] | 1.1484 [1.1406, 1.1553] | 1.1155 |

**Exploratory, after the FAIL: the extra MMA split by GEMM.** This ran on GPU 2
(`results/timing_split_analysis.json`). All arms were exact, and the FC2 positive control changed
the outputs.

| layer | M | both | FC2 (down) only | FC1 (gate/up) only |
|---|---|---|---|---|
| 8 K4 | 1 | 1.0786 | 1.0244 | 1.0779 |
| 8 K4 | 4 | 1.1741 | 1.0247 | 1.1675 |
| 8 K4 | 16 | 1.1498 | 1.0229 | 1.1296 |
| 3 K5 | 1 | 1.0758 | 1.0009 | 1.0748 |
| 3 K5 | 4 | 1.1532 | 1.0240 | 1.1228 |
| 3 K5 | 16 | 1.1189 | 1.0186 | 1.0980 |

The zero-remainder arm is a lower bound on a real remainder's cost. It leaves out the lo-term
quantize, the lo-scale loads and staging the lo tile.

**Side finding: GPU 0 clock pin.**
- Test A ran while GPU 0 held a stale clock pin, flat at 1642 MHz. Its M1 kernel took 108.5 us,
  against 77.8 us on the twin Max-Q GPU 2.
- The pin was reset (`nvidia-smi -rgc`) before any Amendment 2 run.
- The exploratory split on GPU 2 gave ratios similar to Test A's (`results/gpucheck_*.json`).

### 4.2 Test B: two-term carrier, layer level (preregistered). Criteria met

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

\* Arms added after Test A; reported only.
- Pooled down-only: 0.606 [0.549, 0.673].
- Pooled FC1-only: 0.863 [0.842, 0.909].

**Per domain** (two-term ratio): general 0.596, legal 0.509, code/agentic 0.625, reasoning
0.403.

**Like-for-like check.** The 09-04 screen's ordinary-E4M3 geometric-mean NMSE reproduces as
7.3725278e-4 against the published 7.3725270e-4 (relative difference 1e-7). On the same plan,
the two-term carrier gives 2.76e-7, and 16/16 experts improve (`results/testB/screen-repro.json`).

Test B is a layer-level proxy, not KL divergence. Under the preregistration, Test A's failure
stopped the two-hop design regardless of this result.

### 4.3 Test C: H512 before the NVFP4 MLA latent record (preregistered, amended). FAIL

**Mirror check.** On real stand-in latents, the torch mirror matches the real writer's 304-byte
record on 99.96% of elements, and the error energy agrees to 1.6e-5.

| metric | H512 pooled ratio | H512 x random signs |
|---|---|---|
| latent NMSE (unrotated = 0.0090) | 1.0011 [1.0009, 1.0013] | 1.0006 [1.0004, 1.0008] |
| attention-logit error | 1.0055 [1.0044, 1.0066] | 1.0056 [1.0044, 1.0068] |

- The same-layer stand-ins (3:3, 23:23, 43:43) agree within +/-1.6%.
- The latent after `kv_a_layernorm` is already Gaussian-shaped: per-token amax/RMS is 3.25, the
  value for a 512-dimensional Gaussian. There are no outlier channels for a rotation to spread
  (`results/testC/`).

### 4.4 Amendment 2: plane-based down-hop remainder (D-x2)

**K2, real-kernel closure: PASS** (`results/k2/closure-dx2build.json`; reference values from
`results/testB/`).

| layer | path | D_kernel(prod)/D_ref(P-A8) | D-x2/prod (kernel) | reference down-only |
|---|---|---|---|---|
| 3 | M1 (96 tok) | 1.0025 | 0.3650 | 0.3630 |
| 3 | M16/M64/M3072 | 1.0015 | 0.6189 | 0.6183 |
| 8 | M1 (96 tok) | 1.0008 | 0.7844 | 0.7843 |
| 8 | M16/M64/M3072 | 1.0003 | 0.9384 | 0.9384 |

With the flag off, the D-x2 build's outputs are bit-identical to the image's own kernels.

**A2, real-kernel timing: FAIL**, in decode and in prefill. D-x2/prod medians
(`results/analysis_amendment2.json`):

| layer | M1 | M4 | M16 | M64 | M512 | M2048 |
|---|---|---|---|---|---|---|
| 8 K4 | 1.026 | **1.079** | 1.009 | 1.053 | **1.172** | **1.187** |
| 3 K5 | **1.030** | **1.071** | 1.038 | 1.047 | **1.163** | **1.185** |

An exploratory split puts the cost mostly in FC2 (lo staging plus the extra MMA), e.g. +15-16%
at M512. The FC1 epilogue adds 0-3% (`results/dx2_split_timing_raw.json`).

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

The gain on these fresh layers is smaller than on the Test B layers (post-hoc down-only 0.606).
The K4 layers, which have high weight error, gain little.

### 4.5 Amendment 3: row-packed down-hop remainder (D-x2-RP)

**K3, closure: PASS** (`results/k2/closure-rp.json`).
- M1/M16 damage ratios against the reference:
  - layer 3: 0.3650 / 0.6189 (reference 0.3630 / 0.6183);
  - layer 8: 0.7844 / 0.9384 (reference 0.7843 / 0.9384).
- M64 and M3072 outputs are bit-identical to prod.

**A3, timing: PASS.** RP/prod medians over 240 blocks (`results/a3_timing_raw.json`):

| layer | M1 | M4 | M16 |
|---|---|---|---|
| 8 K4 | 1.0254 | 1.0200 | 0.9350 |
| 3 K5 | 1.0009 | 1.0272 [1.016, 1.039] | 1.0030 |

### 4.6 Amendment 4: first production window, NVFP4 MLA KV, 32 windows

The window ran on 2026-09-25 from 19:00:43 to 20:00:59, and production was restored and verified
healthy (`results/kld-rp-20260925/`). The `rp` server logged 168 `P8_DX2_ROWPACK_ACTIVE` lines
(42 layers x 4 ranks); the control server logged 0.

| arm | mean true-decode KLD | window BCa95 |
|---|---|---|
| control | 0.0350129 | [0.02911, 0.04317] |
| rp (D-x2-RP) | 0.0330631 | [0.02801, 0.03983] |

**rp - control = -0.00195 (-5.57%), paired BCa95 [-0.005975, +0.000118], rp lower in 19/32.**
The preregistered reading is **no detectable change**, because the CI crosses 0 by 0.0001.

Post-hoc, reported only (65,472 rows):
- median row KL: -1.5%;
- q90 / q99 / q99.9: -5% / -5% / -8%;
- mean without the top 0.5% of rows: -4.0% (rows above the pooled q99.5 dropped from both arms);
- window medians: lower in 23/32.

Context from the published 09-09 measurements. These are cross-run comparisons, so each carries
server-run noise.
- The 09-25 control vs the published 09-09 control (0.0354562): -1.25% in the mean, and 3.2%
  per-window SD.
- Paired against TR3 4bpw with the same NVFP4 KV (0.03048):
  - control +0.00453 [+0.00029, +0.01032];
  - rp +0.00258 [-0.00119, +0.00577].

  RP closes 43% of the gap.

**Speed.** Production config with MTP3, private port, cooling gate before every cell, one run per
cell. Values are aggregate decode tokens/s. "C1 0K" means concurrency 1 at context 0.

| cell | control | rp | MTP accept (control / rp) |
|---|---|---|---|
| C1 0K | 184.3 | 191.1 (+3.7%) | 0.436 / 0.553 |
| C4 0K | 300.1 | 301.2 (+0.4%) | 0.550 / 0.554 |
| C1 8K | 184.1 | 193.9 (+5.3%) | 0.538 / 0.610 |
| C4 8K | 300.7 | 299.0 (-0.6%) | 0.669 / 0.656 |

RP is not slower. The C1 gains track higher MTP acceptance in these runs. That could come from
more BF16-like target logits or from different generated text, and single runs cannot tell
which.

### 4.7 Amendment 5: second production window, FP8 MLA KV, 32 windows

The window ran on 2026-09-25 from 20:29:49 to 21:17:30, and production was restored and verified
healthy (`results/kld-fp8-20260925/`).

| arm | mean true-decode KLD | window BCa95 |
|---|---|---|
| control-fp8 | 0.0311958 | [0.02643, 0.03776] |
| rp-fp8 | 0.0309337 | [0.02617, 0.03750] |

**rp-fp8 - control-fp8 = -0.00026 (-0.84%), paired BCa95 [-0.001146, +0.001246], lower in
19/32: no detectable change.**

**Paired comparisons across runs.**
- Each carries about 1-2% server-run noise.
- The interval is for the per-window difference in true-decode KLD, and the change is relative
  to the second arm's mean.

| comparison | change | paired BCa95 | first arm lower in |
|---|---|---|---|
| FP8 KV vs NVFP4 KV, control (both 09-25) | -10.9% | [-0.00898, -0.00171] | 24/32 |
| FP8 KV vs NVFP4 KV, rp (both 09-25) | -6.4% | [-0.00337, -0.00037] | 26/32 |
| rp-fp8 vs control with NVFP4 KV (the production configuration, 09-25) | -11.7% | [-0.00892, -0.00212] | 26/32 |
| rp-fp8 vs TR3 4bpw with NVFP4 KV (09-09) | +1.5% | [-0.00315, +0.00357] | 15/32 |
| rp-fp8 vs TR3 4bpw with FP8 KV (09-09) | +9.7% | [-0.00132, +0.00640] | 13/32 |

control-fp8 vs the published 09-09 TrellisMX FP8 control (0.0319452): -2.3% in the mean.

**FP8-KV speed** (`rp-fp8` only, same configuration and caveats):

| cell | control NVFP4 | rp NVFP4 | rp FP8 | rp FP8 MTP accept |
|---|---|---|---|---|
| C1 0K | 184.3 | 191.1 | 193.0 | 0.528 |
| C4 0K | 300.1 | 301.2 | 293.8 | 0.589 |
| C1 8K | 184.1 | 193.9 | 193.4 | 0.487 |
| C4 8K | 300.7 | 299.0 | 304.0 | 0.613 |

**KV capacity**, from the engine startup logs:
- Production profile (GMU 0.88, 48 seqs): NVFP4 12,581,699 tokens; FP8 8,127,659 tokens (-35%).
- Capture profile (GMU 0.97, 1 seq): NVFP4 31,565,217 tokens; FP8 19,362,962 tokens.

### 4.8 Amendment 6: both-hop row-pack (D-x2-RP2)

**Smoke test** (layer 8, rank 0):
- deterministic and finite;
- at M1/M4/M16, RP2 differs from prod by 3.7-3.9% and from RP by 2.7-2.8%;
- at M64/M512, bit-identical to prod.

**K4 closure: PASS** (`results/k2/closure-rp2.json`; reference values from `results/testB/`).

| layer | path | D_kernel(prod)/D_ref(P-A8) | RP2/prod (kernel) | reference both-hops P-A8x2/P-A8 |
|---|---|---|---|---|
| 3 | M1 (96 tok) | 1.0025 | 0.2531 | 0.2514 |
| 3 | M16 | 1.0015 | 0.5494 | 0.5486 |
| 8 | M1 (96 tok) | 1.0008 | 0.5154 | 0.5150 |
| 8 | M16 | 1.0003 | 0.8689 | 0.8689 |

M64 and M3072 outputs are bit-identical to prod on both layers.

**A4 timing: PASS.** RP2/prod medians over 240 blocks (`results/a4_timing_summary.json`):

| layer | M1 | M4 | M16 |
|---|---|---|---|
| 8 K4 | 1.0263 | 1.0064 [1.0009, 1.0128] | 0.9405 |
| 3 K5 | 1.0273 | 0.9879 [0.9843, 0.9917] | 1.0024 |

The input-hop row-pack adds essentially no cost on top of the down-hop RP. On the fresh B2
layers, the layer-level both-hop damage has a geometric mean of about 0.68 (-32%), against 0.744
for down-only.

### 4.9 Amendment 7: final run, 128 windows, FP8 MLA KV

Production was stopped at 21:25:35 and restored healthy at 00:42:03, after the speed arm
(`results/kld-cf128/window.log`, `restoration.json`). Each arm got one fresh server, in this
order:

| arm | server healthy | capture complete | activation proof |
|---|---|---|---|
| control-fp8 | 21:27:51 | 22:22:59 | 0 `P8_DX2_ROWPACK_ACTIVE` lines |
| rp2-fp8 | 22:26:00 | 23:22:17 | 168 `P8_DX2_ROWPACK_ACTIVE ... input_hop=1` lines |
| tr3-fp8 | 23:25:18 | 00:24:23 | 0 lines (TR3 stack) |

All 128 conditional-fit windows were scored in every arm, with 2,046 scored rows per window.

**Arm means** (true-decode mean KLD, window-level BCa95):

| arm | mean KLD | window BCa95 |
|---|---|---|
| control-fp8 | 0.0342826 | [0.03147, 0.03784] |
| rp2-fp8 | 0.0326041 | [0.02990, 0.03615] |
| tr3-fp8 | 0.0300999 | [0.02771, 0.03298] |

**Preregistered paired primaries** (BCa95, 20,000 resamples, seed 20260902):

| difference | mean | relative | BCa95 (relative) | A lower | CI includes 0 | within 5% | within 10% |
|---|---|---|---|---|---|---|---|
| rp2 - control | -0.001679 | -4.9% | [-0.002496, -0.000603] ([-7.3%, -1.8%]) | 105/128 | no | no | yes |
| rp2 - tr3 | +0.002504 | +8.3% | [+0.000599, +0.004364] ([+2.0%, +14.5%]) | 36/128 | no | no | no |
| control - tr3 | +0.004183 | +13.9% | [+0.002106, +0.006137] ([+7.0%, +20.4%]) | 34/128 | no | no | no |

**Reading, by the preregistered rules.**
- **Primary 1: RP2 improves on the control.** KL falls by 4.9%, the CI excludes 0, and RP2 is
  lower in 105 of 128 windows.
- **Primary 2: RP2 is not tied with TR3.** RP2 is 8.3% above TR3, the CI [+2.0%, +14.5%]
  excludes 0, and it is not equivalent at +/-5% or at +/-10%.
- RP2 closes about 40% of the control's gap to TR3 (0.004183 -> 0.002504).
- An interim look at 00:11 used the first 47 windows (+5.4%, CI [-5.3%, +14.2%]) and could not
  tell RP2 and TR3 apart. The full preregistered set can: on the 96 windows scored for the first
  time, RP2 - TR3 is +9.8% [+2.1%, +16.9%].

**Reported only.**
- **The 96 new windows alone:** rp2 - control -4.1% [-6.6%, -0.7%], lower in 76/96.
- **The original 32 windows:** rp2 - control -7.5% [-12.8%, +2.1%], lower in 29/32; rp2 - tr3
  +3.4% [-9.6%, +13.7%].
- **Run-to-run stability on the original 32 windows:**
  - control-fp8 here vs the FP8 window: 0.031174 vs 0.031196 (-0.07%);
  - tr3-fp8 here vs 09-09: 0.027889 vs 0.028190 (-1.07%).

  Run-level drift is about 1% or less, well below the 8.3% RP2-TR3 gap.
- **Row-level ratios** over all 128 x 2,046 rows:

  | comparison | mean | median | q90 | q99 | q99.9 | pooled-q99.5 trimmed | rows lower | window medians lower |
  |---|---|---|---|---|---|---|---|---|
  | rp2 vs control | -4.9% | -4.1% | -5.2% | -5.5% | -3.9% | -5.1% | 52.9% | 100/128 |
  | rp2 vs tr3 | +8.3% | +12.3% | +9.4% | +6.6% | +9.2% | +8.1% | 46.6% | 33/128 |
  | control vs tr3 | +13.9% | +17.1% | +15.5% | +12.8% | +13.7% | +13.5% | 44.9% | 32/128 |

- **Per-domain means** (not preregistered, no intervals):

  | domain (windows) | control-fp8 | rp2-fp8 | tr3-fp8 |
  |---|---|---|---|
  | general (38) | 0.03624 | 0.03500 | 0.03113 |
  | legal (37) | 0.04187 | 0.03868 | 0.03160 |
  | code / agentic (37) | 0.03038 | 0.02933 | 0.03064 |
  | reasoning / termination (16) | 0.02111 | 0.02044 | 0.02293 |

- **Per-domain paired differences** (reported only, not preregistered). These are 12 intervals
  with no multiplicity correction, and each domain has only 16-38 windows.

  | domain | rp2 - control | rp2 - tr3 | control - tr3 |
  |---|---|---|---|
  | general | -3.4% [-9.2, +4.7], 31/38 lower | +12.4% [+2.5, +22.7], 10/38 | +16.4% [+5.5, +28.5], 9/38 |
  | legal | -7.6% [-10.1, -6.0], 35/37 lower | +22.4% [+13.8, +27.6], 4/37 | +32.5% [+24.1, +38.1], 2/37 |
  | code / agentic | -3.5% [-7.0, +5.1], 30/37 lower | -4.3% [-19.0, +9.4], 13/37 | -0.9% [-15.6, +13.2], 14/37 |
  | reasoning / termination | -3.2% [-10.0, +5.0], 9/16 lower | -10.8% [-31.1, +9.5], 9/16 | -7.9% [-30.9, +7.4], 9/16 |

  TR3's advantage sits in legal and general text. On code and reasoning, RP2 and TR3 cannot be
  told apart. RP2's largest gain over the control is on legal text.
- **KV capacity** in the capture profile (engine startup log): TrellisMX FP8 19,333,333 tokens;
  TR3 FP8 24,950,413 tokens.

**Speed.** `rp2-fp8` ran with the production config and the same cells and cooling gate as the
earlier windows, one run per cell. It is shown next to the earlier runs. Values are aggregate
decode tokens/s, with MTP acceptance in parentheses (`results/kld-cf128/final-report.json`).

| cell | control NVFP4 (first window) | rp NVFP4 (first window) | rp FP8 (second window) | rp2 FP8 (final run) |
|---|---|---|---|---|
| C1 0K | 184.3 (0.436) | 191.1 (0.553) | 193.0 (0.528) | 196.4 (0.446) |
| C4 0K | 300.1 (0.550) | 301.2 (0.554) | 293.8 (0.589) | 297.9 (0.503) |
| C1 8K | 184.1 (0.538) | 193.9 (0.610) | 193.4 (0.487) | 200.7 (0.531) |
| C4 8K | 300.7 (0.669) | 299.0 (0.656) | 304.0 (0.613) | 312.7 (0.608) |

The table shows no end-to-end slowdown from RP2, but it cannot rank the arms:
- these are single unpaired runs from three different windows, with no intervals;
- MTP acceptance varies between runs and moves C1 speed by several percent.

The FP8 serving profile of the `rp2-fp8` speed server holds 8,127,659 tokens of KV cache.

## 5. Limits and scope

- **The served effect is expected to be smaller than the measured one.**
  - The true-decode protocol computes every position, including its KV entries, with M1 decode
    steps.
  - In serving, the prompt is prefilled by the unchanged prefill kernels, so the prompt's hidden
    states and KV entries carry control numerics, and only generated tokens use RP2.
  - The served-model effect was not measured.
- **Decode kernels only.** RP and RP2 change only the direct kernels, which handle M <= 16 tokens
  per step (`p8_native_kernel.py`: `small_m = m <= 16` on the full-coupled path).
  - With MTP3, one sequence verifies 4 tokens per step and four sequences verify 16, so both stay
    on the direct path.
  - Five or more concurrent sequences with MTP3 (20+ tokens per step), and all prompt prefill,
    use the grouped and prefill kernels, which are unchanged.
- **Development data, not qualification.** The 32 windows of the first two production windows
  had been opened before, and the final run adds 96 unopened conditional-fit windows. None of
  this is a measurement on the held-back selection, confirmation or final windows, or an
  independent reproduction.
- **One server per arm.** Each arm got one server preparation, in a fixed order, so the window
  intervals do not include server-run variability.
  - The measured drift is about 1% or less: the control -0.07%, TR3 -1.07%, and the NVFP4-KV
    control -1.25% against 09-09.
  - Cross-run comparisons carry that noise.
- **TR3 is a different system.** The TR3 arm runs its own runtime (r10 image, TP4 with expert
  parallelism, DCP4) with different handling of the non-routed weights. The RP2-TR3 difference is
  a system comparison, not an isolated quantizer comparison.
- **Heavy-tailed KL.** A few rows carry much of the mean: the top 1% of rows held 31% of it in
  the first production window. That is why the primaries use window means with BCa intervals.
- **Per-domain results are exploratory.** They were not preregistered, they have no multiplicity
  correction, and the domains are small (16-38 windows).
- **Layer-level proxies are not KL.**
  - Tests B and B2 measure single-layer routed-output damage, not model output.
  - Test C used stand-in latents, because attention inputs were not captured and the residual
    has four hyper-connection streams.
- **Timing scope.**
  - The timings are isolated MoE-layer microbenchmarks on one GPU, with the rank-0 sidecars of
    two layers (3 and 8).
  - The zero-remainder arm is a lower bound on a real remainder's cost.
  - Test A ran on a clock-pinned GPU (see 4.1).
- **Speed is descriptive.** One run per cell. The MTP-acceptance differences between runs are not
  explained.

## 6. Reproduction

### 6.1 Images

| Role | Identity |
|---|---|
| Serving image (public) | `verdictai/trellismx:glm53-flash-p8-r27-reference-20260909@sha256:ca6b80188dce154b91f49108b7d87792d2ba6328935afc71b44d1c0e6f6a1adf` |
| Its base image, per the published image record | `voipmonitor/vllm:jovian-judgement-community-20260906-r27@sha256:a298fe1cd207eaf97bd2ff2686716ed25b7009c09b36650eba732a4a7dc51512` |
| TrellisMX capture image (local build, not published; the image of the published 09-09 measurement) | image id `sha256:0405a1c0dc128b51069798a5d00b346257bbd006c0deb7e6530a3d005d75de71` |
| TR3 capture image (local build, not published; the image of the published 09-09 TR3 comparison) | `local/tr3-r10:cf32-process-cache-20260909`, image id `sha256:62e069faf47f2d42eae5f2c1677f8730a2f3f93d576301fe1bcc7e55f2fdb673` |

The published 09-09 `comparison.json` files record both capture images and their provenance.

### 6.2 Inputs

- **TrellisMX r27 checkpoint** at the revision in [Sources](#7-sources). Its
  `trellismx-manifest.json` is byte-identical to the one we ran, and the sidecar sha256 values
  in `results/testB/layer-003.json` and `layer-008.json` match the Hub files.
- **NVFP4 carrier** at the pinned revision.
- **TR3 4bpw** at `aba59d2175e1ee2887ae0ae1300ba848b1deed84`.
- **BF16 base model** at `a6c167b62691b2bac901344b65cb651a70f53e43`. Tests B and B2 need only the
  routed-expert shards of the tested layers.
- **Teacher dataset.**
  - Revision `95f4fdd94bf29989db2e0d1054e4931f55edb6aa` holds the routed-block captures.
    `scripts/fetch_fit_capture.py` fetches only the fit windows, by byte range, and checks them
    against the capture manifest.
  - Revision `7c378d5f17dba158c4c803eff27c346dd0615660` holds the conditional-fit teacher logits
    and token arrays.
- **Window lists.** `results/kld-rp-20260925/verified-inputs.json` (32 windows) and
  `results/kld-cf128/verified-inputs.json` (all 128) record, for every window:
  - the token-array and token-value sha256;
  - the teacher-logit file and its sha256;
  - the role manifest and metric hashes.
- **Campaign repository** at the cited commit, for the `glm53_nvfp4` modules.

Local paths in the scripts are placeholders (`<workspace>`, `<home>`, `<data-volume>`,
`<model-volume>`); see `scripts/README.md`.

### 6.3 Commands

1. **Kernel trees.** Apply `patches/*.diff` to the image's b12x, or run the builders
   `scripts/build_zero_remainder_b12x.py`, `scripts/build_dx2_b12x.py`, and
   `DX2_DST=<scratch dir> scripts/build_dx2_rp2_b12x.py`.
2. **Test A.** `scripts/run_timing.sh`, then `python3 scripts/analyze_timing.py`.
3. **Tests B and C.**
   - `scripts/fetch_fit_capture.py <layers>`
   - `scripts/run_testB.py`
   - `scripts/run_testC.py`
   - `python3 scripts/analyze_BC.py`
4. **Closure.**
   - `scripts/dx2_base_outputs.py 3 8`
   - `scripts/run_k2.sh` (K2)
   - `K2_ARMS=prod K2_BUILD=image scripts/run_k2_image.sh`
   - `K2_ARMS=prod,rp K2_BUILD=rp scripts/run_k2.sh` (K3)
   - `B12X_TREE=b12x-dx2rp2 K2_ARMS=prod,rp2 K2_BUILD=rp2 scripts/run_k2.sh` (K4)
5. **Timing.**
   - `scripts/run_a2.sh` (A2)
   - `A2_ARM=rp A2_MS=1,4,16 A2_OUT=/work/results/a3_timing_raw.json scripts/run_a2.sh` (A3)
   - `B12X_TREE=b12x-dx2rp2 A2_ARM=rp2 A2_MS=1,4,16 A2_OUT=/work/results/a4_timing_raw.json scripts/run_a2.sh` (A4)
   - then `python3 scripts/analyze_B2_A2.py` and `python3 scripts/summarize_rowpack_timing.py`.
6. **B2.** `scripts/run_testB2.py`, then `python3 scripts/analyze_B2_A2.py`.
7. **End-to-end windows.**
   - The window scripts are `scripts/window_nvfp4_20260925.py`,
     `scripts/window_fp8_20260925.py` and `scripts/window_cf128_20260925.py`.
   - The 128-window inputs come from `scripts/build_inputs_128.py`, and the final report from
     `scripts/final_cf128_report.py`.
   - They need the 09-09 reference launch arguments, the capture images above and, for the final
     run, the TR3 attempt-2 launch.
   - They also operate our production host (systemd user unit, locks, ports). Treat them as a
     record of the procedure and adapt them before use.

### 6.4 Re-deriving the reported numbers (no GPU)

- **Timing and layer-level analyses.** `analyze_timing.py`, `analyze_timing_split.py`,
  `analyze_BC.py` and `analyze_B2_A2.py` regenerate `timing_analysis.json`,
  `timing_split_analysis.json`, `analysis_BC.json` and `analysis_amendment2.json` exactly from
  the raw files in `results/`.
- **A3 and A4.** `summarize_rowpack_timing.py` reproduces the A3 and A4 medians and intervals.
- **KL divergence.** Every production window includes three kinds of score file:
  - `scores/*.json`, the per-window records. Each window's `true_decode_mean_kld` is the mean of
    `kld[1:]` in the matching `.npz`.
  - `scores/*.npz`, the per-row scores: row KL, teacher entropy, top-1 tokens and
    probabilities, and realized-token log-probabilities.
  - `scores/*.retirement.json`, hash records that tie each score file to its retired raw logits.
- **Intervals.** The intervals in each `analysis.json` are `scipy.stats.bootstrap` BCa over the
  window-paired values. The cross-run comparisons use the same estimator against the published
  09-09 `comparison.json` files.

**Not included:** `.pt` tensors; raw logits, which were retired after hashing and scoring;
request/response captures; server and startup logs; telemetry; clock CSVs.

## 7. Sources

### 7.1 Models, data, code and images used

1. **Base model.** `zai-org/GLM-5.3-Flash-BF16`, revision
   `a6c167b62691b2bac901344b65cb651a70f53e43` (https://huggingface.co/zai-org/GLM-5.3-Flash-BF16).
   It is the model under study: its logits are the KL teacher, and its experts are the reference
   for the layer tests.
2. **NVFP4 carrier.** `local-inference-lab/GLM-5.3-Flash-NVFP4`, revision
   `520de24eabf507659eaef7c70f14fd584527facc`
   (https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4). It supplies every non-routed
   weight of the TrellisMX runtime (the checkpoint manifest pins this revision) and the attention
   weights used in Test C.
3. **TrellisMX r27 checkpoint.** `brandonmusic/GLM-5.3-Flash-TrellisMX-MXFP8`, revision
   `b492969185c600f2dc431fe12acbbdf2632e905b`
   (https://huggingface.co/brandonmusic/GLM-5.3-Flash-TrellisMX-MXFP8). It is the checkpoint under
   test, and that revision holds the published 09-09 KL results we compare against
   (`results/kld-reference-20260909/`, `results/kld-tr3-20260909/`).
4. **TR3 4bpw.** `brandonmusic/GLM-5.3-Flash-tr3-4bpw`, revision
   `aba59d2175e1ee2887ae0ae1300ba848b1deed84`
   (https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw). It is the comparator of the
   final run and of the 09-09 comparison, and it was re-captured in the same capture image.
5. **Teacher logits and captures.** Dataset `brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits`
   (https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits), revisions
   `95f4fdd94bf29989db2e0d1054e4931f55edb6aa` and `7c378d5f17dba158c4c803eff27c346dd0615660`. It
   holds the BF16 routed-block captures used by the layer tests, and the teacher logits and token
   arrays scored in every window.
6. **Campaign repository.** https://github.com/brandonmmusic-max/glm53-hadamard-shapleymcg-kld,
   commit `03ff56b20a5934748ca5117010c24aad9c208536`.
   - Its `glm53_nvfp4` modules are reused unchanged:
     - `p8_layer_rate_damage`, `p8_coupled_scale`, `capture`, `shard_index`;
     - `canary_mxfp6_reap` (`_qdq_e4m3_k32`, `_metrics`);
     - `screen_p8_h128_activation` (`_ordinary`);
     - `p8_decode_protocol`, the KL capture and scoring protocol, which was imported from a
       local checkout of the same repository and is byte-identical to the file at this commit.
   - Its `results/P8_COUPLED_INCOHERENCE_ARCHIVE_AUDIT.md` holds the published 09-04 screen
     value that Test B reproduces.
7. **Serving image.** `verdictai/trellismx:glm53-flash-p8-r27-reference-20260909` (digest above).
   Every kernel test and production arm ran in it.
8. **b12x.** Luke Alonso and contributors, https://github.com/local-inference-lab/b12x (the
   earlier https://github.com/lukealonso/b12x redirects there). Apache License 2.0.
   - The P8 kernels and the patched files come from it.
   - The 16 existing files that the patches modify are identical to commit
     `d564f6ca54c092497ec5ae7e07a272f55eec7dbe` of the TrellisMX integration branch
     (https://github.com/brandonmmusic-max/b12x).
9. **Decode benchmark.** `llm_decode_bench.py` version 0.4.29 from
   https://github.com/local-inference-lab/llm-inference-bench (the script's own update URL). The
   sha256 of the copy we ran is
   `7239958032ec10781d7db3efca23a7602bd9aeb94455a723e2634ab7d6fe546a`. It produced every speed
   number.

### 7.2 Background

- **QTIP.** Tseng, A., Sun, Q., Hou, D., and De Sa, C. "QTIP: Quantization with Trellises and
  Incoherence Processing." NeurIPS 2024. arXiv:2406.11235. Cited as the origin of the trellis
  codes with computed codebooks that the P8 weight format uses; the b12x intrinsics label the
  trellis path "QTIP/EXL3".
- **ExLlamaV3 / EXL3.** turboderp. https://github.com/turboderp-org/exllamav3. Cited because the
  P8 decode's procedural MCG constants and state construction are ported from ExLlamaV3's MCG
  decoder, as the b12x source headers state.
- **Microscaling.** Open Compute Project, "OCP Microscaling Formats (MX) Specification v1.0,"
  2023; Rouhani, B. D., et al. "Microscaling Data Formats for Deep Learning." arXiv:2310.10537,
  2023. Cited because they define the E4M3 elements with UE8M0 scales per 32-element block that
  the P8 activations and the remainder use.
- **NVIDIA PTX ISA.** https://docs.nvidia.com/cuda/parallel-thread-execution/. Cited for the
  block-scaled `mma.sync` (`kind::mxf8f6f4`) that the kernels issue, and for the NVFP4 format
  (E2M1 values with E4M3 scales) of the KV record.
- **QuaRot.** Ashkboos, S., Mohtashami, A., Croci, M., Li, B., Jaggi, M., Alistarh, D.,
  Hoefler, T., and Hensman, J. "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs." NeurIPS
  2024. arXiv:2404.00456. Cited as the rotation-before-quantization idea that Test C tested on the
  KV record.
- **KL divergence.** Kullback, S., and Leibler, R. A. "On Information and Sufficiency." Annals of
  Mathematical Statistics 22(1), 1951. Cited because it defines the primary metric.
- **BCa intervals.** Efron, B. "Better Bootstrap Confidence Intervals." Journal of the American
  Statistical Association 82(397), 1987. Cited because every KL interval is a BCa interval.
- **SciPy.** `scipy.stats.bootstrap` (`method="BCa"`). Cited because it computes the KL intervals.
- **Preregistration.** Nosek, B. A., Ebersole, C. R., DeHaven, A. C., and Mellor, D. T. "The
  preregistration revolution." PNAS 115(11), 2018. Cited for the practice this study follows:
  a plan and decision rules fixed before the data, with amendments stated openly.

## 8. License

This repository's own scripts, documentation and result files are distributed under the
ShapleyMCG License 1.0 (`LICENSE`), the same license as the campaign repository. It is
source-available, not OSI open source. Required attribution:

> ShapleyMCG was created by Brandon M. Music. Canonical source:
> https://github.com/brandonmmusic-max/shapleymcg

The b12x kernel patches in `patches/` modify Apache-2.0 b12x sources and are provided under the
Apache License 2.0 (`patches/LICENSE.b12x`). Third-party models, datasets, images and libraries
keep their own licenses.

## 9. Repository layout

```
README.md                 this document
PREREG.md                 preregistration and Amendments 1-7
RESULTS.md                results log, written as results came in
LICENSE                   ShapleyMCG License 1.0
scripts/                  every script that produced a number (see scripts/README.md)
patches/                  b12x kernel diffs and their Apache-2.0 license (see patches/README.md)
results/
  timing_raw.json, timing_analysis.json                Test A
  timing_split_raw.json, timing_split_analysis.json    Test A exploratory split
  gpucheck_*.json                                      per-GPU M1 timing and clock checks
  testB/, testB2/                                      layer-level records, 09-04 screen reproduction
  testC/                                               Test C per-configuration records
  analysis_BC.json, analysis_amendment2.json           Tests B, C, B2 and A2 analysis
  k2/closure-*.json                                    K2, K3 and K4 closure
  a2_timing_raw.json, a3_timing_raw.json               A2 and A3 timing
  a4_timing_raw.json, a4_timing_summary.json           A4 timing
  dx2_split_timing_raw.json                            A2 exploratory split
  kld-rp-20260925/                                     first production window (NVFP4 KV, 32 windows)
  kld-fp8-20260925/                                    second production window (FP8 KV, 32 windows)
  kld-cf128/                                           final run (FP8 KV, 128 windows):
    analysis.json, final-report.json                     preregistered analysis and final report
    verified-inputs.json                                 window list with input and teacher hashes
    control-fp8/, rp2-fp8/, tr3-fp8/                     runtime audits and per-window scores
    speed/rp2-fp8/                                       benchmark outputs and launch arguments
    window.log, execution*.json, restoration.json        window log, execution and restoration records
```
