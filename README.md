# FP8 remainder carriers and NVFP4-KV rotation on TrellisMX P8 kernels (GLM-5.3-Flash)

Brandon M. Music, September 2026

**Status: complete (2026-09-26). Corrected after an independent audit the same day (section 6).
The RP2 kernels were then released in a public image (section 7).**

This repository is the full record of one preregistered study:
- the preregistration, its seven amendments and a chronology of when each was committed
  (`PREREG.md`);
- the results log, written as results came in (`RESULTS.md`);
- every script that produced a number (`scripts/`; the release speed harnesses sit next to their
  results in `results/speed-20260926/`);
- the raw and analysed result files (`results/`);
- the kernel patches (`patches/`);
- the deployment files and image record of the RP2 release image (`deploy/`).

This README explains what was tested, why, how, and what came out, including every test that
failed. In sections 1-6, "production" means the serving configuration during the study: the r27
reference image with NVFP4 KV.

## Summary

We serve GLM-5.3-Flash from our TrellisMX r27 checkpoint on custom mixture-of-experts kernels
called "P8". The routed-expert weights are trellis-coded, and the kernel decodes them to 8-bit
floating point (E4M3) inside the matrix multiply. Activations are rounded to E4M3, with one
power-of-two scale per 32 values.

The study asked two questions:
- Can a second FP8 "remainder" term for the activations bring the model measurably closer to its
  BF16 original without slowing decode?
- Can a fixed rotation make the 4-bit (NVFP4) attention KV cache more accurate?

The findings, in the order the tests ran:

1. **A second FP8 multiply per decoded weight fragment is not free at decode.** It cost 7-17% of
   MoE-layer time. (Test A: FAIL.)
2. **At layer level, two-term activations remove about half the error.** Carrying activations as
   two FP8 terms (hi + lo) removes 51.0% of the routed-output error against the BF16 experts.
   (Test B.)
3. **A fixed 512-point Hadamard rotation before the NVFP4 MLA KV cache gave no improvement on the
   tested stand-in latents.** (Test C: FAIL.)
4. **A second data plane for the remainder is correct but too slow.** Adding the remainder only at
   the down-projection input, in a second data plane, reproduces the reference damage (closure
   PASS) but costs up to 19% of layer time (timing FAIL). (Amendment 2.)
5. **Row-packing brings the kernel cost down to at most 2.7% of MoE-layer time at M1.** The
   remainder goes into a row of the tensor-core tile that the decode kernels already compute and
   throw away. This was done first at the down-projection input (RP), then at both inputs (RP2),
   and every gate passed. (Amendments 3 and 6.)
6. **The final, preregistered end-to-end run** used all 128 conditional-fit windows with an FP8 KV
   cache (Amendment 7).
   - **RP2 improves on the production kernels.** It lowers true-decode KL divergence from the
     BF16 teacher by **4.9%** (95% CI -7.3% to -1.8%), the preregistered improvement reading.
   - **RP2 stays 8.3% above TR3 4bpw** (+2.0% to +14.5%). Equivalence within +/-5% or +/-10% was
     not established.
   - **The TR3 gap depends on the text mix.** TR3 leads on legal and general text. With the four
     domains weighted equally:
     - rp2 - tr3 is +6.1% [-0.2%, +12.5%], an interval that includes 0;
     - RP2 vs the control is -4.7% [-7.3%, -2.0%].
   - **RP2 closes 40.1% of the control's gap to TR3** (95% [18.6%, 75.0%]).
7. **The KV cache mattered more.**
   - Switching the attention KV cache from NVFP4 to FP8 lowered KL divergence by 10.9% for the
     control, at a cost of 35% of KV capacity. That comparison is cross-run and was not a
     preregistered primary.
   - The earlier 32-window checks of the down-hop-only RP were inconclusive: -5.6% with NVFP4 KV
     and -0.8% with FP8 KV, and both intervals crossed 0.
8. **Release, after the study.** The RP2 kernels ship in a public image with FP8 KV as the
   default, together with the host-side guards and EPI-PAR, a parallel version of the M1 epilogue
   (exploratory, section 4.10).
   - A pre-flight check found the image's kernels bit-identical to the measured ones in 56 of 56
     cells. The 128-window KL result carries over to the image through that bounded check; the
     image itself was not re-measured.
   - In same-window single-run A/Bs against the r27 reference configuration (NVFP4 KV), the
     release configuration was -2.0% to +12.0% in decode and +5.7% to +7.5% in prefill. These
     runs cannot separate the kernels from the KV cache or from MTP acceptance (section 7).
   - Raising GPU 3's power limit from 300 W to 350 W afterwards made prefill on the production
     server 6-8% faster (section 7.4).

**Scope.**
- **KV cache.** RP2 was KL-measured only with FP8 KV. Production used NVFP4 KV during the study,
  and RP2 was never run with it.
- **Kernel paths.** RP and RP2 change only the direct decode kernels. Dispatch is by tokens per
  step: steps with M <= 16 tokens use them.
  - The KL measurement pushes every token through those kernels.
  - In serving, steps with more than 16 tokens, which covers most prompt prefills, use unchanged
    kernels, so the served effect is expected to be smaller. That was not measured.
- **Run-to-run stability.** Server-level shifts between repeat runs on different days reached
  -2.4%, with CIs that exclude 0. That is about the size of the primaries' 1.8-2.0% margins from
  0. The arms of the final run ran in one night, in a fixed order, so the conclusions assume
  within-night stability. The one same-night repeat (-0.07%) supports that assumption.

| Question | Test | Outcome |
|---|---|---|
| Is a second FP8 MMA per decoded weight fragment free at decode? | Test A: zero-remainder timing | **FAIL.** Median time ratio 1.0698 / 1.0854 at M1 and 1.1650 / 1.1484 at M4 (layers 8 / 3); the bar was <= 1.03. |
| How much routed-output damage does a two-term (hi + lo) FP8 activation carrier remove? | Test B: layer level, 6 layers | Pooled damage ratio 0.490 (-51.0%), BCa 95% CI [0.427, 0.562]; 6/6 layers improve. |
| Does a fixed H512 rotation before the NVFP4 MLA latent record reduce error? | Test C: 6 MLA layers, stand-in latents | **FAIL.** No improvement on the tested stand-in latents: pooled ratio 1.0011 (latent NMSE) and 1.0055 (attention-logit error). |
| Down-hop remainder in a second data plane (D-x2) | K2 / A2 / B2 | Closure PASS. Timing FAIL: 1.071-1.079 at M4, 1.163-1.187 in prefill. Fresh-layer damage 0.744 (-25.6%), B2 PASS. |
| Down-hop remainder packed into the idle MMA row +8 (D-x2-RP) | K3 / A3 | Closure PASS. Timing PASS: median ratio <= 1.0272 at M1 and M4. |
| End-to-end KLD, NVFP4 MLA KV, 32 windows | Amendment 4 | control 0.0350129 vs rp 0.0330631: -5.57%, paired BCa 95% [-0.005975, +0.000118]: **no detectable change**. |
| End-to-end KLD, FP8 MLA KV, 32 windows | Amendment 5 | control-fp8 0.0311958 vs rp-fp8 0.0309337: -0.84%, [-0.001146, +0.001246]: **no detectable change**. Cross-run, reported only: FP8 vs NVFP4 KV for the control -10.9% [-0.00898, -0.00171]. |
| Remainder packed at both hops (D-x2-RP2) | K4 / A4 | Closure PASS. Timing PASS: median ratio <= 1.0273 at M1 and M4. |
| **Final: 128 windows, FP8 MLA KV** | Amendment 7 | rp2 - control **-4.9% [-7.3%, -1.8%]**, lower in 105/128: **improvement**. rp2 - TR3 **+8.3% [+2.0%, +14.5%]**: RP2 above TR3, equivalence within +/-5% or +/-10% not established; with equal domain weights +6.1% [-0.2%, +12.5%]. |
| Parallel M1 epilogue for RP2 (EPI-PAR; exploratory, not preregistered) | EPI-PAR check | Identity PASS, 28/28 rows in both builds. M1-only build: rp2+EPI-PAR/rp2 0.9693 (layer 8) and 0.9726 (layer 3). |
| Do the release image's kernels reproduce the measured ones? | Release pre-flight | 56/56 cells bit-identical. |

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
  kept, still left 13-59% of the routed-output error on the tested layers (the "activation share"
  column below).
- Carrying the activations as two FP8 terms, a value plus its rounded remainder, attacks that
  share directly.
- The open question was cost. On a trellis kernel, decoding the weights is the expensive step, so
  reusing a decoded weight fragment for a second multiply might be close to free. Test A measured
  this, and it was not free.
- The row-pack variants (RP and RP2) then placed the second term where the decode kernels already
  spend the work. That brought the kernel cost to at most 2.7% of MoE-layer time at M1.

**A rotation for the NVFP4 KV cache.**
- Production stores the attention cache as a 4-bit NVFP4 record.
- A fixed orthogonal rotation before quantization spreads outlier channels across the vector, as
  in QuaRot. Its inverse could be folded into the attention weights, so it would cost nothing at
  runtime.
- On the tested stand-in latents the rotation found nothing to spread. Their Gaussian shape is a
  likely explanation, not a demonstrated cause, and confirmation on real latents was not done.

## 2. Background

### 2.1 The P8 kernels

In the TrellisMX r27 checkpoint, the routed experts of GLM-5.3-Flash's 42 MoE layers are trellis
bitstreams at 4 or 5 bits per weight ("K4" or "K5"), chosen per layer. All other weights come
from an NVFP4 carrier checkpoint.

The P8 kernels live in b12x:
- **Weights.** Each weight fragment is decoded in registers to E4M3 with a procedural MCG codebook.
  Its constants and state construction are ported from ExLlamaV3's procedural MCG decoder, and
  that family of trellis codes descends from QTIP. The decoded fragment goes straight into a
  block-scaled tensor-core matrix multiply ("MMA": `mma.sync`, `kind::mxf8f6f4`, shape m16n8k32).
- **Activations.** E4M3 values with a power-of-two UE8M0 scale per 32 elements along the reduction
  dimension ("E4M3/UE8M0-K32", the OCP Microscaling layout). They are quantized at two points,
  called "hops":
  - the FC1 (gate/up projection) input;
  - the FC2 (down projection) input after SwiGLU.
- **Coupled transform.** The checkpoint's block Hadamard transforms and per-channel scales sit
  around both multiplies.
- **Kernel paths.** "M" below is the number of tokens in one MoE-layer call, that is, tokens per
  step.
  - M <= 16 uses the "direct" small-M kernels.
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
row (`valid_rows == 1`). A route row is one token sent to one expert. The other rows of the A
operand are zeros that the tensor core multiplies anyway. In the m16n8 accumulator layout, each
thread holds the same output columns for rows q and q+8.

**D-x2-RP** (the down-projection remainder) uses that spare row:
- **FC1 epilogue.** Writes `lo = q(v - dq(hi))`, with its own UE8M0 byte, into row +8 of the
  route's own tile.
- **FC2 main loop.** Unchanged: it already stages and multiplies row 8, and its odd lanes already
  supply the row-8 scale.
- **FC2 epilogue.** Folds row 8 into row 0 in registers before the FP16 store
  (`fragment[0] += fragment[2]`, `fragment[1] += fragment[3]`).

**D-x2-RP2** adds the same trick at the FC1 input:
- the input prologue writes `x_lo = q(x - dq(x_hi))` into token row +16;
- FC1 stages it into A row 8, in place of the broadcast duplicate that row used to hold;
- the FC1 epilogue folds the row q+8 accumulators into row q.

Neither variant adds an MMA, a staging pass or a second data plane. Steps with M > 16 keep the
production single-term kernels, so both variants act only on steps with at most 16 tokens. In
serving, those are mostly decode steps.

**Supported configuration.** The design relies on three settings, all of which the measured runs
used:
- `fc1_broadcast_a=True` (for RP2);
- `fc1_warp_quant=False` (for all remainder modes);
- one route per direct tile.

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
- A preregistered FAIL stops its direction. Test A's FAIL ended the two-hop design whatever Test B
  showed, and the down-hop work restarted under Amendment 2 on fresh data.

**Chronology.** These are the workspace git commits that added each part, against the first data
each part governs.

| Part | Commit | Committed (EDT) | First data it governs |
|---|---|---|---|
| Preregistration (Tests A-C) | f0a5d99 | 2026-09-23 12:13:31 | Test A/B/C results committed 12:52:51 (0f64f6b) |
| Amendment 1: Test C moved to the model's actual MLA layers (only every fourth layer has the MLA cache; the rest use KDA linear attention); records that Test A had already failed | f527d0d | 2026-09-23 12:39:48 | Test C results, 12:52:51 |
| Amendment 2: down-hop remainder confirmation on fresh layers (B2) and on the real kernel (K2, A2) | 61757d2 | 2026-09-23 17:36:07 | K2 log written 17:49:02; A2 log 18:03:01 |
| Amendment 3: row-packed down-hop remainder (K3, A3) | 0d7a0e6 | 2026-09-23 18:13:05 | K3 log 18:17:32; A3 log 18:21:04 |
| Amendment 4: decode-scored KLD with the 09-09 protocol, NVFP4 KV, first production window | caa6c2b | 2026-09-25 18:47:54 | window 1 started 19:00:43 |
| Amendment 5: second window with FP8 KV, then the FC1-input row-pack | 3ec4dab | 2026-09-25 20:26:35 | window 2 started 20:29:49 |
| Amendment 6: gates for the both-hop row-pack RP2 (K4, A4) | 7622e63 | 2026-09-25 20:36:09 | RP2 smoke 21:18; K4 and A4 started 21:19 |
| Amendment 7: final run on all 128 conditional-fit windows with FP8 KV (arms `control-fp8`, `rp2-fp8`, `tr3-fp8`; two primaries; an `rp2-fp8` speed arm) | 724782a | 2026-09-25 21:23:44 | window 3 started 21:25:35 |

- **Git times are self-asserted** local timestamps, not an independent registration service.
- **This public repository cannot show the order.** It was created after the first two
  production windows.
- **Header corrections.** The approximate times in the headers of Amendments 5-7 (~20:10, ~20:40,
  ~22:00) are corrected in the table in `PREREG.md`, for example Amendment 7 to 21:23:44. The
  amendment texts are unchanged.
- **Amendment 3** was written after the Amendment 2 timing results and in response to them, before
  any D-x2-RP number.

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
| K2 / K3 / K4 | Per layer and path: `abs(D_kernel(prod) / D_ref(P-A8) - 1) <= 5%`, and the arm/prod damage ratio within +/-0.05 of the reference ratio (down-only for K2/K3, both hops for K4). K3 and K4 also require M64/M3072 outputs bit-identical to prod. |
| A2 | Decode: median <= 1.03 at M1 and M4. Prefill: median <= 1.05 at M512 and M2048. |
| A3 / A4 | Median <= 1.03 at M1 and M4 on both layers. |
| Amendments 4 and 5 reading | Improvement if the paired mean < 0 and the CI excludes 0. No detectable change if the CI includes 0. Harm if the mean > 0 and the CI excludes 0. |
| Amendment 7 primaries | (1) `rp2-fp8 - control-fp8`: improvement if the mean < 0 and the CI excludes 0. (2) `rp2-fp8 - tr3-fp8`, reported three ways: whether the CI includes 0, and whether the whole CI lies within +/-5% or within +/-10% of TR3's mean. |

### 3.3 Layer-level numerics (Tests B and B2)

**What.** `scripts/remainder_damage.py` uses the campaign's `p8_layer_rate_damage` method:
- the r27 rank sidecars are reference-decoded with the kernel's procedural-MCG E4M3 codebook;
- the exact coupled reference forward runs over all 288 experts;
- the routed-output damage is `D = sum_t ||S(t)||^2 / sum_t ||routed_output(t)||^2`, measured
  against the BF16 source experts on 64 fit windows x 48 tokens = 3,072 tokens per layer.

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
- A like-for-like rerun of an earlier (09-04) activation screen reproduces its published value to
  a relative difference of 1e-7, so the harness measures the same thing the earlier work did.

**Uncertainty.** A paired window BCa bootstrap: one resample of the 64 window indices is applied to
every layer at once (20,000 replicates, seed 20260923, jackknife acceleration;
`scripts/analyze_BC.py`). The statistic is the mean over layers of
`log(sum_w D_new / sum_w D_ref)`, and its exponential is the pooled geometric-mean ratio.

### 3.4 KV rotation (Test C)

**What.**
- **Stand-in latents.** Attention inputs are not in the capture, so the latent for MLA layer A is
  built from the routed-block capture of layer C:
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
- All four tensor-parallel rank sidecars run one at a time, and their outputs are summed, which is
  what the all-reduce computes.
- The Test B fit tokens are fed in chunks that exercise every kernel path: M1, M16, M64 and M3072.
  The M1 path used the first 96 tokens; the preregistration said 64.
- Damage is measured against the BF16-source routed output.
- K4 uses the RP2 tree (`B12X_TREE=b12x-dx2rp2`) and the `rp2` arm.

**Why.**
- The layer-level numbers come from a reference implementation. Closure checks that the CUDA
  kernels reproduce the reference damage, so the reference numbers carry over to the kernels. It
  shows damage agreement, not tensor equivalence.
- It catches layout and scale bugs that a speed test cannot see.
- Untouched paths are checked for bit identity with `torch.equal`, on the saved rank-summed
  tensors, to show that the new code does not leak into the production kernels.

### 3.6 Timing with CUDA graphs, interleaved blocks and paired ratios

**What.** `scripts/timing_harness.py` and `scripts/dx2_timing.py`.
- `P8NativeTPMoE` is built for tensor-parallel rank 0 the way the serving integration builds it,
  with real routed-block inputs and top-8 routes.
- Each timed unit is a CUDA-graph replay of one MoE-layer call, 200 replays per block.
- The two arms alternate block by block: 60 blocks per cell. Decode cells use 4 input sets (240
  blocks) and prefill cells use 2.
- The metric is the median of per-block paired time ratios, with a percentile-bootstrap interval of
  that median (20,000 resamples, seed 20260923).
- Compile caches were off.

**Why.**
- Graph replay removes Python and launch overhead, so only kernel time is measured.
- Alternating the arms puts both under the same clock and temperature conditions, so drift cancels
  in the ratio.
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
- RP and RP2 change only the direct kernels, which run the steps with at most 16 tokens. A
  prefill-scored KL measurement would compute each 2,048-token window in prompt-prefill steps of
  far more than 16 tokens. Those run the unchanged kernels, so it could not see RP or RP2.
- Forcing the tokens keeps every arm on an identical history, so rows line up exactly across arms
  and the comparison is row for row.
- Reusing the published protocol makes our controls comparable with the published 09-09 values.

**Consequence.** Every position, including the KV entries it writes, is computed by M1 decode
steps. In serving, prompt-prefill steps with more than 16 tokens use the unchanged kernels
(section 5).

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
through per-window differences.
- **Estimator.** Intervals come from `scipy.stats.bootstrap` with `method="BCa"`, 20,000 resamples
  and `random_state=20260902`, the estimator of the published intervals.
- **Relative intervals.** A relative interval is the absolute BCa interval divided by the
  comparator's observed mean.
- **Endpoint reproducibility.** With a fixed seed, the endpoints depend on the window order. The
  manifest order in `verified-inputs.json` reproduces them exactly.
- **Software.** Python 3.12.3, numpy 1.26.3, scipy 1.16.3.

**Why.**
- Rows within a window share context and are strongly correlated, so treating 2,046 rows as
  independent would overstate precision. The window is the resampling unit.
- Window means span more than an order of magnitude (`results/kld-cf128/analysis.json`). Pairing
  removes that difficulty spread from the comparison.
- KL is heavy-tailed: in the first production window, the top 1% of rows carried 31% of the mean.
  BCa corrects the bias and skew of the bootstrap distribution better than a plain percentile
  interval.

**Window dependence.** Windows are not fully independent. The reasoning windows share a template,
and a few general-text windows overlap. A cluster bootstrap over windows that share text keeps
both primaries' intervals clear of 0 (section 4.9).

### 3.10 Only conditional-fit windows

**What.** The teacher dataset assigns its windows to roles: fit, conditional-fit, selection,
confirmation and final.
- This study read only fit windows (layer tests) and conditional-fit windows (KL runs).
- The first two production windows reused 32 conditional-fit windows that the 09-09 measurements
  had already opened.
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
- The FP8-KV window showed that FP8 KV lowers KL by 10.9% for our control, in a cross-run
  comparison.
- The published 09-09 values show the same direction for TR3: 0.0281899 with FP8 KV, against
  0.03048 with NVFP4 KV.
- The final comparison is therefore made at each system's better cache. The cost is that RP2 was
  never measured with the NVFP4 KV that production used during the study.

### 3.12 Fixed arm order, and what repeat runs show

**What.** One production window and one fresh server per arm, in a fixed order: control, then RP2,
then TR3.

**Why.** Fresh servers rule out state carried from one arm to the next. A fixed order could
confound a server-level shift with the arm, so repeats of identical configurations were compared
on the original 32 windows (paired BCa95, relative to the earlier run; `reported-extras.json`):

| Repeat | Mean shift | BCa95 | Per-window SD | Max per-window |
|---|---|---|---|---|
| control-fp8, final vs FP8 window (same night, about 1 h apart) | -0.07% | [-0.9%, +1.2%] | 3.6% | 12.3% |
| control-fp8, FP8 window vs 09-09 | -2.3% | [-7.4%, -0.2%] | 5.2% | 19.4% |
| control-fp8, final vs 09-09 | -2.4% | [-7.7%, -0.3%] | 5.7% | 20.0% |
| control NVFP4, first window vs 09-09 | -1.3% | [-2.4%, -0.1%] | 3.2% | 9.5% |
| tr3-fp8, final vs 09-09 | -1.1% | [-5.3%, +0.3%] | 3.8% | 16.2% |

- **Per-window noise** is already inside the paired intervals, because each arm is one run. A shift
  of the whole server run is not.
- **Same night vs different days.** The same-night repeat agrees closely. Repeats on different days
  moved by up to -2.4%, with CIs that exclude 0.
- **Margins.** The primaries clear 0 by 1.8% (rp2 - control) and 2.0% (rp2 - tr3), about the size
  of those cross-day shifts.
- **Assumption.** The conclusions therefore assume that server shifts within one night are small.
  The one same-night repeat supports this.

### 3.13 Estimators and seeds

| Analysis | Estimator | Resamples | Seed |
|---|---|---|---|
| Test A, exploratory split, A2, A3 and A4 timing | percentile bootstrap of the median over blocks | 20,000 | 20260923 |
| Tests B, C and B2 | paired window BCa with jackknife acceleration | 20,000 | 20260923 |
| Test C random-sign arm | fixed random signs | n/a | 20260923 |
| End-to-end KLD, including cross-run comparisons and repeats | `scipy.stats.bootstrap`, BCa, over window-paired differences; relative intervals scaled by the comparator's observed mean | 20,000 | 20260902 |
| Audit sensitivities: domain-balanced (stratified), cluster bootstrap, gap fraction | percentile bootstrap over windows or clusters | 20,000 | 20260902 |
| Audit sensitivities: Bonferroni-adjusted primaries (97.5%), ratio of means | `scipy.stats.bootstrap`, BCa, over window-paired values | 20,000 | 20260902 |
| EPI-PAR timing (exploratory) | median of per-block ratios per input set, then the median over sets; no interval | n/a | n/a |
| Release speed A/Bs | single runs; no interval | n/a | n/a |

### 3.14 Model, checkpoint, runtime and hardware

- **Base model.** `zai-org/GLM-5.3-Flash-BF16`. It is the KL teacher and the reference for the
  layer-level damage measurements.
- **Checkpoint.** TrellisMX r27 (`brandonmusic/GLM-5.3-Flash-TrellisMX-MXFP8`): per-rank trellis
  sidecars for the routed experts, four tensor-parallel ranks per layer, overlaid on the NVFP4
  carrier `local-inference-lab/GLM-5.3-Flash-NVFP4`.
- **Runtime.** Serving image `verdictai/trellismx@sha256:ca6b8018...` (full digest in section 8.1).
  It is vLLM-based, and the server reports
  `0.26.1rc0+glm53.flash.nvfp4.luke.clean.r1.vllme75bcfd.b12x58a046f`. Every patched arm
  bind-mounts a patched b12x tree over `/opt/glm53-flash/b12x`.
- **Capture images.** The KL captures ran in capture-only images (section 8.1).
- **Serving layout.**
  - Capture profile: TP4/DCP4 (tensor parallel 4, decode context parallel 4), MTP off, one
    sequence, 4096 batched tokens, GPU memory utilization 0.97, maximum length 1M.
  - Production speed profile: MTP with 3 speculative tokens, 48 sequences, 8192 batched tokens,
    GPU memory utilization 0.88.
- **Hardware.** Four NVIDIA RTX PRO 6000 Blackwell GPUs (two Max-Q, two Workstation Edition), PCIe,
  300 W limit each during the study. GPU 3's limit was raised to 350 W afterwards (section 7.4).

## 4. Results

Every number below comes from `RESULTS.md` or the files under `results/`, copied at the precision
of the source.

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

**Per domain** (two-term ratio): general 0.596, legal 0.509, code/agentic 0.625, reasoning 0.403.

**Like-for-like check.** The 09-04 screen's ordinary-E4M3 geometric-mean NMSE reproduces as
7.3725278e-4 against the published 7.3725270e-4 (relative difference 1e-7). On the same plan, the
two-term carrier gives 2.76e-7, and 16/16 experts improve (`results/testB/screen-repro.json`).

Test B is a layer-level proxy, not KL divergence. Under the preregistration, Test A's failure
stopped the two-hop design regardless of this result.

### 4.3 Test C: H512 before the NVFP4 MLA latent record (preregistered, amended). FAIL

**Mirror check.** On real stand-in latents, the torch mirror matches the real writer's 304-byte
record on 99.96% of elements, and the error energy agrees to 1.6e-5. A rerun from the saved
tensors gives 99.963% of elements equal and an error-energy ratio of 0.999984
(`results/testC/mirror_check_20260926.txt`).

| metric | H512 pooled ratio | H512 x random signs |
|---|---|---|
| latent NMSE (unrotated = 0.0090) | 1.0011 [1.0009, 1.0013] | 1.0006 [1.0004, 1.0008] |
| attention-logit error | 1.0055 [1.0044, 1.0066] | 1.0056 [1.0044, 1.0068] |

- The same-layer stand-ins (3:3, 23:23, 43:43) agree within +/-1.6%.
- On the tested stand-in latents, the rotation found nothing to spread. After `kv_a_layernorm` the
  per-token amax/RMS is 3.25, the value for a 512-dimensional Gaussian (`results/testC/`).
- The Gaussian shape is a likely explanation, not a demonstrated cause. Confirmation on real
  latents was not done.

### 4.4 Amendment 2: plane-based down-hop remainder (D-x2)

**K2, real-kernel closure: PASS** (`results/k2/closure-dx2build.json`; reference values from
`results/testB/`). K2 shows damage agreement with the reference.

| layer | path | D_kernel(prod)/D_ref(P-A8) | D-x2/prod (kernel) | reference down-only |
|---|---|---|---|---|
| 3 | M1 (96 tok) | 1.0025 | 0.3650 | 0.3630 |
| 3 | M16/M64/M3072 | 1.0015 | 0.6189 | 0.6183 |
| 8 | M1 (96 tok) | 1.0008 | 0.7844 | 0.7843 |
| 8 | M16/M64/M3072 | 1.0003 | 0.9384 | 0.9384 |

With the flag off, the D-x2 build's outputs are bit-identical to the image's own kernels. This was
confirmed by `torch.equal` on the saved tensors (M16 and M3072, layers 3 and 8, rank sums).

**A2, real-kernel timing: FAIL**, in decode and in prefill. D-x2/prod medians
(`results/analysis_amendment2.json`):

| layer | M1 | M4 | M16 | M64 | M512 | M2048 |
|---|---|---|---|---|---|---|
| 8 K4 | 1.026 | **1.079** | 1.009 | 1.053 | **1.172** | **1.187** |
| 3 K5 | **1.030** | **1.071** | 1.038 | 1.047 | **1.163** | **1.185** |

An exploratory split puts the cost mostly in FC2 (lo staging plus the extra MMA), e.g. +15-16% at
M512. The FC1 epilogue adds 0-3% (`results/dx2_split_timing_raw.json`).

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
- The damage ratios agree with the plane-based D-x2 to 4-5 digits.
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

**rp - control = -0.00195 (-5.57%), paired BCa95 [-0.005975, +0.000118], rp lower in 19/32.** The
preregistered reading is **no detectable change**, because the CI crosses 0 by 0.0001.

Post-hoc, reported only (65,472 rows):
- median row KL: -1.5%;
- q90 / q99 / q99.9: -5% / -5% / -8%;
- mean without the top 0.5% of rows: -4.0% (rows above the pooled q99.5 dropped from both arms);
- window medians: lower in 23/32.

Context from the published 09-09 measurements. These are cross-run comparisons, so each carries
server-run noise (section 3.12).
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

In these single runs rp was not slower. The C1 gains track higher MTP acceptance, which could come
from more BF16-like target logits or from different generated text; single runs cannot tell which.

### 4.7 Amendment 5: second production window, FP8 MLA KV, 32 windows

The window ran on 2026-09-25 from 20:29:49 to 21:17:30, and production was restored and verified
healthy (`results/kld-fp8-20260925/`).

| arm | mean true-decode KLD | window BCa95 |
|---|---|---|
| control-fp8 | 0.0311958 | [0.02643, 0.03776] |
| rp-fp8 | 0.0309337 | [0.02617, 0.03750] |

**rp-fp8 - control-fp8 = -0.00026 (-0.84%), paired BCa95 [-0.001146, +0.001246], lower in 19/32:
no detectable change.**

**Paired comparisons across runs.** These were reported, not preregistered primaries, and each
carries server-run noise (section 3.12). The interval is for the per-window difference in
true-decode KLD, and the change is relative to the second arm's mean.

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

**KV capacity**, from the engine startup logs (`results/kv_capacity_serving_profile.json` for the
serving profile):
- Production profile (GMU 0.88, 48 seqs): NVFP4 12,581,699 tokens, measured on the first window's
  control speed server; FP8 8,127,659 tokens (-35%).
- Capture profile (GMU 0.97, 1 seq): NVFP4 31,565,217 tokens; FP8 19,362,962 tokens.

### 4.8 Amendment 6: both-hop row-pack (D-x2-RP2)

**Smoke test** (layer 8, rank 0; `results/rp2_smoke_20260925_console.jsonl`):
- deterministic and finite;
- at M1/M4/M16, RP2 differs from prod by 3.7-3.9% and from RP by 2.7-2.8%;
- at M64/M512, bit-identical to prod.

**K4 closure: PASS** (`results/k2/closure-rp2.json`; reference values from `results/testB/`). K4
shows damage agreement with the reference.

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

RP2's kernel cost at M1 is at most 2.7% of MoE-layer time, and the input-hop row-pack adds
essentially nothing on top of the down-hop RP. On the fresh B2 layers, the layer-level both-hop
damage has a geometric mean of about 0.68 (-32%), against 0.744 for down-only.

### 4.9 Amendment 7: final run, 128 windows, FP8 MLA KV

Production was stopped at 21:25:35 and restored healthy at 00:42:03, after the speed arm
(`results/kld-cf128/window.log`, `restoration.json`). Each arm got one fresh server, in this order:

| arm | capture image (from `launch.json`) | server healthy | capture complete | activation proof |
|---|---|---|---|---|
| control-fp8 | `sha256:0405a1c0...` | 21:27:51 | 22:22:59 | 0 `P8_DX2_ROWPACK_ACTIVE` lines |
| rp2-fp8 | `sha256:0405a1c0...` | 22:26:00 | 23:22:17 | 168 `P8_DX2_ROWPACK_ACTIVE ... input_hop=1` lines |
| tr3-fp8 | `sha256:62e069fa...` | 23:25:18 | 00:24:23 | 0 lines (TR3 stack) |

- **Image fields.** The `capture_image_id` field in each `runtime-audit.json` is a harness constant
  and is wrong for the TR3 arm; that file's `image` field is right.
- **Activation marker.** `P8_DX2_ROWPACK_ACTIVE` is printed when the MoE runtime is constructed,
  meaning the path is armed; it runs on steps with M <= 16.
- **Coverage.** All 128 conditional-fit windows were scored in every arm, with 2,046 scored rows
  per window.

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
- **Primary 2: RP2 is not tied with TR3.** RP2 is 8.3% above TR3 and the CI [+2.0%, +14.5%]
  excludes 0. Equivalence within +/-5% or +/-10% was not established. That is a separate
  statement from the direction: it does not show that the difference exceeds 10%.
- **Gap to TR3.** RP2 closes 40.1% of the control's gap to TR3 (0.004183 -> 0.002504), paired
  bootstrap 95% [18.6%, 75.0%].
- **Interim look.** There was one interim look, at 23:47 EDT, when control and rp2 were complete
  and TR3 was at 47 windows. On those windows rp2 - tr3 was +5.4% [-5.3%, +14.2%], and they could
  not tell RP2 and TR3 apart. The look changed nothing in the run, which completed automatically.
- **New windows.** The full preregistered set does separate them. On the 96 windows scored for
  the first time, rp2 - tr3 is +9.8% [+2.1%, +16.9%].

**Reported only.**
- **The 96 new windows alone:** rp2 - control -4.1% [-6.6%, -0.7%], lower in 76/96.
- **The original 32 windows:** rp2 - control -7.5% [-12.8%, +2.1%], lower in 29/32; rp2 - tr3
  +3.4% [-9.6%, +13.7%].
- **Run-to-run repeats:** see the table in section 3.12. The same-night control repeat moved
  -0.07%; repeats across days moved by up to -2.4%.
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

- **Per-domain paired differences** (reported only, not preregistered). These are 12 intervals with
  no multiplicity correction, and each domain has only 16-38 windows.

  | domain | rp2 - control | rp2 - tr3 | control - tr3 |
  |---|---|---|---|
  | general | -3.4% [-9.2, +4.7], 31/38 lower | +12.4% [+2.5, +22.7], 10/38 | +16.4% [+5.5, +28.5], 9/38 |
  | legal | -7.6% [-10.1, -6.0], 35/37 lower | +22.4% [+13.8, +27.6], 4/37 | +32.5% [+24.1, +38.1], 2/37 |
  | code / agentic | -3.5% [-7.0, +5.1], 30/37 lower | -4.3% [-19.0, +9.4], 13/37 | -0.9% [-15.6, +13.2], 14/37 |
  | reasoning / termination | -3.2% [-10.0, +5.0], 9/16 lower | -10.8% [-31.1, +9.5], 9/16 | -7.9% [-30.9, +7.4], 9/16 |

  TR3's advantage sits in legal and general text. On code and reasoning, RP2 and TR3 cannot be told
  apart. RP2's largest gain over the control is on legal text.
- **KV capacity** in the capture profile (engine startup log): TrellisMX FP8 19,333,333 tokens;
  TR3 FP8 24,950,413 tokens.

**Audit sensitivity analyses** (reported only; `results/kld-cf128/reported-extras.json` unless
noted).

| Analysis | rp2 - control | rp2 - tr3 |
|---|---|---|
| Preregistered (BCa95; `analysis.json`) | -4.9% [-7.3%, -1.8%] | +8.3% [+2.0%, +14.5%] |
| Domain-balanced: four domains weighted equally, stratified bootstrap (percentile 95%) | -4.7% [-7.3%, -2.0%] | +6.1% [-0.2%, +12.5%] |
| Cluster bootstrap, windows sharing >= 1 32-token span (81 clusters, largest 19) | [-7.4%, -1.9%] | [+2.0%, +16.2%] |
| Cluster bootstrap, windows sharing >= 64 spans (110 clusters, largest 16) | [-7.7%, -2.1%] | [+1.7%, +15.8%] |
| Bonferroni-adjusted 97.5% BCa | [-7.59%, -1.22%] | [+1.08%, +15.31%] |
| Ratio-of-means BCa (paired, denominator resampled too) | [-7.20%, -1.80%] | [+1.87%, +14.57%] |

- **Robustness.** RP2's advantage over the control holds under every analysis. Its gap to TR3
  holds under all of them except equal domain weights, where the interval includes 0.
- **Window dependence.** 256 window pairs share at least one 32-token span, and all 120 pairs of
  reasoning windows share 100 or more (a common template).
- **Per-window spread.** The rp2/control ratio ranges from 0.56 to 2.89 across windows (median
  0.93; log-ratio SD 0.18), so a single window says little.
- **Rerun agreement.** The two control-fp8 runs on the original 32 windows are not bit-identical:
  - 0/32 windows are identical;
  - 74.9% of rows differ;
  - 97.7% of top-1 predictions agree between runs.
- **Top-1 agreement with the BF16 teacher**, over all rows: control 94.34%, rp2 94.47%, tr3 94.75%.

**Speed.** `rp2-fp8` ran with the production config and the same cells and cooling gate as the
earlier windows, one run per cell. It is shown next to the earlier runs. Values are aggregate
decode tokens/s, with MTP acceptance in parentheses (`results/kld-cf128/final-report.json`).

| cell | control NVFP4 (first window) | rp NVFP4 (first window) | rp FP8 (second window) | rp2 FP8 (final run) |
|---|---|---|---|---|
| C1 0K | 184.3 (0.436) | 191.1 (0.553) | 193.0 (0.528) | 196.4 (0.446) |
| C4 0K | 300.1 (0.550) | 301.2 (0.554) | 293.8 (0.589) | 297.9 (0.503) |
| C1 8K | 184.1 (0.538) | 193.9 (0.610) | 193.4 (0.487) | 200.7 (0.531) |
| C4 8K | 300.7 (0.669) | 299.0 (0.656) | 304.0 (0.613) | 312.7 (0.608) |

Observed sustained-decode rates were similar in these single unpaired runs. The runs cannot rank
the arms or resolve an end-to-end cost of 1-3%. The FP8 serving profile of the `rp2-fp8` speed
server holds 8,127,659 tokens of KV cache.

### 4.10 Exploratory, after Amendment 7: EPI-PAR (not preregistered)

EPI-PAR is a speed change to the M1 FC1 epilogue that must leave every output bit-identical, with
or without RP2. It was not preregistered and is not part of any KL measurement. It is reported
here because the release image ships it (section 7).

**Idea.** In the direct FC1 kernels, the down-input quantization runs one thread per row.
- A direct tile holds one route, so a single thread walks the row's four blocks of 32 values one
  after another. With RP or RP2 it also runs the lo pass for each block, while the rest of the
  thread block waits at the closing barrier.
- EPI-PAR (`scripts/build_rp2_epipar_b12x.py`, `patches/b12x-epipar-m1.diff`) gives each
  (row, 32-block) pair its own thread.
- The per-block code is the serial path's text with the thread index replaced by the row index.
  The loads, the max order, the quantizer and the stores are the same.
- FC1 and FC2 are separate kernel launches, so the extra writing threads need no new
  synchronization.
- It is switched on by `B12X_P8_EPI_PAR=1`, which is read when the MoE runtime is constructed and
  joins the compile spec.

**Check** (`scripts/epipar_check.py`, `scripts/run_epipar_check.sh`; GPU 1, next to the idle
production server):
- four arms: prod, prod + EPI-PAR, rp2, rp2 + EPI-PAR;
- layers 8 (K4) and 3 (K5), with fit-capture routes; M1, M4 and M16 with 4 input sets each, plus
  M64 and M512;
- identity is gated first. Timing then runs 60 CUDA-graph blocks x 200 replays per arm, with the
  arm order rotated every block.

**Identity: PASS in both builds, 28/28 rows** (`results/epipar_check_both.json`,
`results/epipar_check_k5.json`):
- prod + EPI-PAR equals prod, and rp2 + EPI-PAR equals rp2, bit for bit (`torch.equal`) on every
  M1, M4 and M16 input set;
- M64 and M512 are identical across all four arms;
- rp2 + EPI-PAR is deterministic.

**Timing** (median over the 4 input sets of the per-set medians of per-block ratios; descriptive):

| build | cell | rp2/prod | rp2+EPI-PAR/prod | rp2+EPI-PAR/rp2 |
|---|---|---|---|---|
| EPI-PAR in the M1 and M2-16 kernels | L8 M1 | 1.0317 | 1.0000 | 0.9693 |
| | L3 M1 | 1.0602 | 1.0305 | 0.9715 |
| | L8 M16 | 0.9763 | 1.0231 | **1.0479** (slower) |
| | L3 M16 | 1.0015 | 1.0034 | 1.0018 |
| EPI-PAR in the M1 kernel only (`EPIPAR_FILES=route_hoist_k5.py`) | L8 M1 | 1.0321 | 1.0004 | 0.9693 |
| | L3 M1 | 1.0597 | 1.0312 | 0.9726 |
| | L8 M4 | 1.0165 | 1.0118 | 0.9864 (identical code) |
| | L3 M4 | 1.0345 | 1.0268 | 0.9811 (identical code) |
| | L8 M16 | 0.9759 | 0.9768 | 1.0009 (identical code) |

**Reading (exploratory).**
- In the M1 kernel, EPI-PAR removed RP2's whole M1 cost on the K4 layer and about half of it on
  the K5 layer, with identical outputs.
- In the M2-16 kernel it made L8 M16 about 5% slower. The M1-only build was therefore kept, and it
  is the one in the release image.
- In the M1-only build, cells whose code is identical in both arms differ by up to 1.9%. That is
  the observed timing variation between identical-code cells in this run.
- This run measured RP2's own M1 cost at 1.032 (L8) and 1.060 (L3), above A4's 1.026 and 1.027.
  The conditions differ: GPU 1 next to the resident production server here, and four interleaved
  arms instead of two.
- EPI-PAR's end-to-end effect was not measured on its own.

## 5. Limits and scope

- **FP8 KV only.** RP2 was KL-measured only with FP8 KV. Production used NVFP4 KV during the
  study, and RP2 was never run with NVFP4 KV.
- **Served effect.** The served effect is expected to be smaller than the measured one:
  - the true-decode protocol computes every position, including its KV entries, with M1 decode
    steps;
  - in serving, dispatch is by tokens per step. Steps with at most 16 tokens run RP2, and
    prompt-prefill steps with more than 16 tokens use the unchanged kernels, so most of a prompt's
    hidden states and KV entries carry control numerics.

  The served-model effect was not measured.
- **Direct kernels only.** RP and RP2 change only the direct kernels, which handle steps with
  M <= 16 tokens (`p8_native_kernel.py`: `small_m = m <= 16` on the full-coupled path).
  - With MTP3, one sequence verifies 4 tokens per step and four sequences verify 16, so both stay
    on the direct path.
  - Five or more concurrent sequences with MTP3 (20+ tokens per step), and steps with more than 16
    tokens (most prompt prefills), use the grouped and prefill kernels. Those are unchanged.
- **The activation marker means the path is armed.** `P8_DX2_ROWPACK_ACTIVE` is printed at
  construction. The row-packed path then runs on steps with M <= 16.
- **Configuration limits.** The remainder code supports only the configuration the measured runs
  used:
  - `fc1_broadcast_a=True` for RP2;
  - `fc1_warp_quant=False` for every remainder mode;
  - one route per direct tile.

  Other settings could race or silently drop the remainder. The builder now adds host-side
  checks that reject them by default, and the release image has them; `DX2_GUARDS=0` builds the
  measured tree without them. `results/rp2_guard_smoke.json` records that:
  - the measured configuration passes, with smoke numbers identical to the 09-25 run;
  - the four unsupported combinations tested are rejected.
- **Server-level shifts.** Each arm got one server preparation.
  - Repeats of identical configurations on different days moved by up to -2.4%, with CIs that
    exclude 0. That is about the size of the primaries' margins from 0 (1.8% and 2.0%).
  - The final arms ran in one night, in a fixed order, and the conclusions assume within-night
    stability. The one same-night repeat (-0.07% [-0.9%, +1.2%]) supports that.
  - Cross-run comparisons carry the larger cross-day noise.
- **Captures are not bit-reproducible run to run.** Two control-fp8 runs had 0/32 identical
  windows. That row-level noise is part of the per-window variation the paired intervals already
  include.
- **Window dependence and domain mix.**
  - Windows are not fully independent: the reasoning windows share a template, and a few
    general-text windows overlap. The cluster bootstrap leaves both primaries intact.
  - The RP2-TR3 gap depends on the text mix. With equal domain weights its interval includes 0.
- **Development data, not qualification.**
  - The 32 windows of the first two production windows had been opened before, and the final run
    adds 96 unopened conditional-fit windows.
  - None of this is a measurement on the held-back selection, confirmation or final windows.
  - None of it is an independent reproduction.
- **TR3 is a different system.** The TR3 arm runs its own runtime (r10 image, TP4 with expert
  parallelism, DCP4) with different handling of the non-routed weights. The RP2-TR3 difference is
  a system comparison, not an isolated quantizer comparison.
- **Heavy-tailed KL.** A few rows carry much of the mean: the top 1% of rows held 31% of it in the
  first production window. That is why the primaries use window means with BCa intervals.
- **Per-domain results are exploratory.** They were not preregistered, they have no multiplicity
  correction, and the domains are small (16-38 windows).
- **Layer-level proxies are not KL.**
  - Tests B and B2 measure single-layer routed-output damage, not model output.
  - The closure gates show damage agreement, not tensor equivalence.
  - Test C used stand-in latents, because attention inputs were not captured and the residual has
    four hyper-connection streams.
- **Timing scope.**
  - The timings are isolated MoE-layer microbenchmarks on one GPU, with the rank-0 sidecars of two
    layers (3 and 8).
  - The zero-remainder arm is a lower bound on a real remainder's cost.
  - Test A ran on a clock-pinned GPU (see 4.1).
- **Speed is descriptive.** One run per cell, here and in the release A/Bs of section 7. The runs
  cannot resolve an end-to-end cost of 1-3%, and the MTP-acceptance differences between runs are
  not explained.

## 6. Independent audit (2026-09-26)

After the first publication (commit 05b5d28), the repository and the testing workspace were
audited by six independent reviews.
- **Who.** The reviews were run as separate automated review agents: three on the local TrellisMX
  model and three on two hosted models.
- **How.** Each built its own reading of this repository and the workspace, under read-only rules.
  In a second round each cross-checked the others' findings against the evidence.

**Verdict.** All six concluded "supported with changes", and none found a blocker to the numbers:
- all 384 score records and hashes reproduce;
- both preregistered primaries reproduce exactly, including their BCa endpoints in manifest order;
- the layer-level, closure and timing tables regenerate.

**What changed after the audit:**
- **Wording.**
  - Equivalence with TR3 "was not established", rather than "not equivalent".
  - Test C is limited to the tested stand-in latents.
  - "Nearly free" is replaced by the measured kernel cost (at most 2.7% at M1).
  - The closure gates are described as damage agreement.
  - Prefill coverage is stated by tokens per step.
  - The speed claim now says the runs cannot resolve a 1-3% end-to-end cost.
- **Drift.** "About 1% or less" is replaced by the run-to-run repeat table (section 3.12), and the
  within-night stability assumption is stated.
- **New sensitivity analyses** (section 4.9):
  - domain-balanced;
  - cluster bootstrap for windows that share text;
  - Bonferroni;
  - ratio-of-means;
  - an interval for the gap closure;
  - per-window spread;
  - rerun row agreement;
  - top-1 agreement with the teacher.

  They come from `scripts/reported_extras_cf128.py`.
- **Provenance.**
  - A chronology of amendment commits against first data (section 3.1).
  - The interim-look time corrected to 23:47.
  - Per-arm capture images from each arm's `launch.json`, with the misleading `capture_image_id`
    field flagged.
  - The start-state `execution.json` records of the first two windows.
- **Kernel safety.** Host-side guards (`patches/b12x-dx2-guards.diff`) and a guard smoke test.
  They were first opt-in and are on by default since the third round (below).
- **Records added:**
  - the 09-25 smoke output;
  - the KV-mirror check rerun;
  - the serving-profile KV capacities;
  - copies of the published 09-09 comparison files;
  - a working repo-layout path in `scripts/final_cf128_report.py`.
- **Number fixes.**
  - One rounding fix in `RESULTS.md`: the new-96 control - tr3 lower bound is +6.6%, not +6.7%.
  - The NVFP4 serving-capacity figure is attributed to the correct server.

**Third round (sign-off, on commit 4bfd853).** Five of the six reviews checked the corrected
repository. Two signed off, and one returned no usable report. Two did not sign, pending the
first four fixes below; the fifth answers a smaller note from the same round. This update makes
all five, and it has not been re-reviewed:
- **Wording.** `RESULTS.md` now marks the Test C "Why it fails" paragraph as a likely
  explanation, not a demonstrated cause, and corrects its prefill wording in place. This README
  states prefill coverage by tokens per step throughout.
- **Attribution.** `scripts/reported_extras_cf128.py` now computes the Bonferroni and
  ratio-of-means intervals it was credited with, and they are in `reported-extras.json`. The
  interval description in section 9.2 now separates the preregistered BCa intervals from the
  percentile-bootstrap sensitivity analyses.
- **Guards by default.** The RP2 builder adds the host guards unless `DX2_GUARDS=0`, which
  reproduces the measured trees byte for byte.
- **Exit status.** `scripts/rp2_guard_smoke.py` exits nonzero when a check fails.
- **Patch notes.** `patches/README.md` states which patches stack.

**Fourth round (release sign-off, on commit 068f51c).** Five reviews checked the release
materials of section 7 and the model card. One signed off. Four asked for fixes; the model-card
fixes were made on the card, and this update makes the rest (not re-reviewed):
- `scripts/epipar_check.py` and `scripts/final_identity_check.py` exit nonzero when a check
  fails.
- The EPI-PAR reading calls the 1.9% spread the observed timing variation between identical-code
  cells, not a noise floor.
- Section 3.7 states the prefill argument by tokens per step.
- Section 7 states how the KL result carries over to the release image, records how the DCP4
  window's zero clock offsets are known, and marks the underfilled DCP4 cell in the table.
- The image record's `kernel_source` names all three patches in order.

## 7. RP2 release image (2026-09-26)

After the study and the audit, the RP2 kernels were released as a public image. Nothing in this
section is preregistered. The image deploys the measured configuration, a pre-flight check
compares its kernels with the measured ones, and the speed A/Bs are descriptive single runs.

### 7.1 The image

`verdictai/trellismx:glm53-flash-p8-r27-rp2-20260926@sha256:a0392e1c370eb933d87511928ea3aba6d5d63965cfe6eeaf9cc026a667e98ddc`
(Docker Hub)

- **Contents.** The September 9 r27 reference image plus one layer (`deploy/Dockerfile`). The
  layer replaces `/opt/glm53-flash/b12x` with the release b12x tree and adds the launcher
  `serve-rp2.sh`.
- **Kernel tree.** RP2, the host-side guards, and EPI-PAR in the M1 FC1 kernel (section 4.10).
  The tree equals the image's b12x with `b12x-dx2-rowpack-rp2.diff`, `b12x-dx2-guards.diff` and
  `b12x-epipar-m1.diff` applied in that order. `scripts/build_rp2_epipar_b12x.py` with
  `EPIPAR_FILES=route_hoist_k5.py` builds the same tree (section 8.3).
- **Defaults.** `B12X_P8_DOWN_REMAINDER=rp2`, `B12X_P8_EPI_PAR=1` and FP8 MLA KV
  (`KV_CACHE_DTYPE=fp8`). FP8 KV is the default because the final KL result was measured with it.
  Setting both B12X variables to empty runs the reference kernels from the same image.
- **Requirements.** RP2 needs `GLM53_P8_FC1_BROADCAST_A=1` and `GLM53_P8_FC1_WARP_QUANT=0`, the
  defaults. `serve-rp2.sh` and the kernel guards reject other settings, and the launcher accepts
  only TP4/DCP4.
- **Deployment files** (`deploy/`):
  - `compose.yaml`, the serving profile: TP4/DCP4, MTP3, 48 sequences, 8,192 batched tokens,
    GMU 0.88;
  - `serve-rp2.sh` and the `Dockerfile`;
  - `image-record-rp2-20260926.json`, the image record. Its sha256 values for the other three
    files match the copies here.
- **Relation to the measured arm.** The 128-window KLD (section 4.9) was measured on the
  `b12x-dx2rp2` tree mounted in the capture image, without guards and without EPI-PAR.
  - The guards are host-side checks and change no kernel code.
  - EPI-PAR is designed to be bit-identical, and the pre-flight below found it so.
  - The KLD result carries over to the release image through that bounded identity check (56/56
    tested cells, section 7.2). The release image itself was not re-measured.
- **Production.** Production switched to this image with FP8 KV at the end of the DCP4 speed
  window below (`results/speed-20260926/dcp4/window.log`).

### 7.2 Pre-flight: kernel identity, 56/56 cells

Scripts `final_identity_check.py` and `run_final_identity.sh`; result
`results/final_identity_check.json`.
1. **Save.** The r27 reference image ran with the measured RP2 tree (`b12x-dx2rp2`) mounted. It
   saved the prod (all flags off) and rp2 outputs for real fit-capture routes
   (`final_identity_save.log`).
2. **Compare.** The release image ran with its own b12x. Four arms were rebuilt and compared with
   `torch.equal` (`final_identity_compare.log`):
   - rp2 + EPI-PAR (the image default) and rp2, against the saved rp2 outputs;
   - prod and prod + EPI-PAR, against the saved prod outputs.
3. **Coverage.** Layers 3, 8, 23 and 43; TP ranks 0 and 3; M1 (4 input sets), M4, M16 and M64.
   All 56 cells were bit-identical. RP2 differs from prod in the 48 cells with M <= 16 and matches
   it at M64, as designed.

- **What "prod" means here.** The saved prod outputs come from the measured tree with all flags
  off. That this path equals the unpatched image's own kernels follows from the flag gating
  (`patches/README.md`). It was checked directly only for the earlier D-x2 build (K2: M16 and
  M3072).
- **Which image.** The pre-flight ran on the release image by its tag on the build host, before
  the push. The DCP4 speed window recorded the local id of the image it ran as
  `sha256:a0392e1c...`, the pushed digest (`results/speed-20260926/dcp4/candidate/image.json`).
  The DCP1 launch records name the image by tag only.
- **Bounded.** The check covers the tested layers, ranks and M paths. It supports identity on
  those cells, not a proof for every input.

### 7.3 Speed: two same-window A/Bs

Both A/Bs compare the release configuration (RP2 image, FP8 KV) with the r27 reference
configuration (reference image, NVFP4 KV).
- Each arm got one fresh server and one run per cell, in a fixed order.
- Values are tokens/s, aggregate over concurrent requests (C1 is one request, C8 is eight), with
  MTP acceptance in parentheses.
- `results/speed-20260926/speed-summary-20260926.json` collects both tables and their conditions.

**DCP1, greedy fixed-input method.** TP4/DCP1, MTP3, 24 sequences, 4,096 batched tokens, GMU 0.97.
Measured 2026-09-26 between 12:31 (first server start) and 13:04, RP2 arm first
(`results/speed-20260926/dcp1/`).
- **Decode.** `bench_fixed.py` runs the unchanged `llm_decode_bench.py` v0.4.29 with a fixed
  run-ID prompt prefix and temperature 0. Exact token targeting, 20 s cells, at most 8,192
  tokens.
- **Prefill.** `llm_decode_bench.py --prefill-only`, cold, 20 s per context.
- **Harness.** The first RP2 cell (C1 0K) ran under `run.py`, with the cooling gate described for
  DCP4 below, and its log ends after that cell. The remaining cells of both arms ran with
  `quick.sh`, which waits up to 60 s for the hottest GPU to reach 55 °C and then proceeds.
- **Servers.** `serve-rp2.sh` accepts only DCP4, so both DCP1 servers started the image's own
  serving script directly, with the settings above (`dcp1/rp2/launch.json`,
  `dcp1/control-launch.json`).

| cell | r27 reference, NVFP4 KV | RP2 image, FP8 KV | RP2 / reference |
|---|---|---|---|
| C1 decode, 0K | 198.9 (0.51) | 222.7 (0.59) | 1.120 |
| C1 decode, 8K | 224.4 (0.55) | 229.4 (0.58) | 1.022 |
| C1 decode, 32K | 218.0 (0.59) | 226.7 (0.52) | 1.040 |
| C1 decode, 128K | 210.5 (0.57) | 210.1 (0.52) | 0.998 |
| prefill, 8K | 8,613 | 9,255 | 1.075 |
| prefill, 32K | 8,584 | 9,111 | 1.061 |
| prefill, 128K | 8,199 | 8,709 | 1.062 |

**DCP4, production serving profile, default sampling.** TP4/DCP4, MTP3, 48 sequences, 8,192
batched tokens, GMU 0.88. Measured 2026-09-26, 05:28-06:42, reference arm first
(`results/speed-20260926/dcp4/`).
- **Decode.** `llm_decode_bench.py` v0.4.29 with the server's default sampling. 30 s cells, at
  most 2,048 tokens.
- **Prefill.** `--prefill-only`, cold, 20 s per context.
- **Cooling gate** before every cell: at least 90 s idle, and all GPUs at or below 55 °C for 30 s.
- **Values** are from `summary.json`, which `summarize.py` regenerates byte for byte from the raw
  files.

| cell | r27 reference, NVFP4 KV | RP2 image, FP8 KV | RP2 / reference |
|---|---|---|---|
| C8 decode, 0K | 581.5 (0.58) | 569.9 (0.50) | 0.980 |
| C8 decode, 8K | 478.1 (0.62) | 499.3 (0.59) | 1.044 |
| C8 decode, 16K | 469.7 (0.51) | 474.5 (0.65) | 1.010 |
| C8 decode, 32K | 471.0 (0.63), underfilled* | 473.9 (0.60) | 1.006 |
| C1 decode, 0K | 186.7 (0.60) | 195.0 (0.48) | 1.044 |
| C1 decode, 8K | 188.7 (0.66) | 197.8 (0.57) | 1.049 |
| C1 decode, 16K | 184.6 (0.47) | 190.3 (0.79) | 1.031 |
| C1 decode, 32K | 185.3 (0.35) | 185.4 (0.40) | 1.001 |
| C1 decode, 64K | 187.9 (0.53) | 193.0 (0.63) | 1.027 |
| C1 decode, 128K | 181.9 (0.56) | 183.6 (0.65) | 1.009 |
| prefill, 8K | 8,102 | 8,618 | 1.064 |
| prefill, 16K | 8,265 | 8,732 | 1.057 |
| prefill, 32K | 8,292 | 8,775 | 1.058 |
| prefill, 64K | 8,219 | 8,716 | 1.060 |
| prefill, 128K | 8,072 | 8,559 | 1.060 |

\* The benchmark flagged this control cell as underfilled (`underfilled: true` in
`dcp4/control/decode-ctx32768-c8.json`): 7.8 requests running on average instead of 8.

**Reading.**
- **Three changes at once.** Each A/B changes the kernels (RP2 with EPI-PAR), the KV cache (FP8
  against NVFP4) and the image layer together. It cannot attribute a difference to any one of
  them.
- **MTP acceptance moves decode speed.** With three draft tokens, a step emits on average 1 + 3a
  tokens, where a is the acceptance rate. At equal step time, the DCP1 C1 0K acceptance difference
  alone (0.506 against 0.590) predicts a 10.0% speed difference, most of the 12.0% observed.
  Acceptance differs between the arms in every cell, in both directions.
- **Prefill** was 5.7-7.5% faster in the release configuration, in every cell of both tables.
  Prefill steps with more than 16 tokens run the unchanged MoE kernels, so this is consistent with
  the FP8 KV cache. The runs did not isolate the cause.
- **Do not mix the tables.** They differ in method (greedy fixed inputs against default sampling;
  20 s against 30 s cells; at most 8,192 against 2,048 tokens), in DCP and scheduler profile, and
  in GPU clock offsets (below). Sampling changes the generated text, and with it MTP acceptance.
  The September 9 figures published with the r27 image used the greedy fixed-input method at 24
  sequences, 4,096 batched tokens and GMU 0.97, with DCP4. In method they compare only with the
  DCP1 table, which differs from them in DCP.
- **Hardware.** Four RTX PRO 6000 Blackwell GPUs at a 300 W power limit each, with a +6000 memory
  clock offset. GPU3, a 600 W-class card held to 300 W, runs at lower sustained clocks than its
  twin, GPU1, under the same load and settings. GPU3's limit was raised to 350 W after these A/Bs
  (section 7.4).
  - **DCP1.** Both arms ran with a +150 MHz GPC clock offset on GPU3 (`dcp1/run.log` and the
    conditions in `speed-summary-20260926.json`).
  - **DCP4.** No clock offsets were applied during the window. The window itself did not record
    clock state. An NVML read after the window, before any clock change, showed a GPC offset of 0
    on every GPU.

**KV capacity** (engine startup logs).
- **Serving profile** (DCP4, GMU 0.88, 48 sequences): FP8 KV holds 8,127,659 tokens, against
  12,581,699 with NVFP4 KV (-35%; section 4.7, `results/kv_capacity_serving_profile.json`). The
  servers of the DCP4 A/B reported 8,141,843 and 12,575,163 (`runtime-audit.json`).
- **DCP1** at GMU 0.97: the RP2 image with FP8 KV held 3,823,412 tokens (`dcp1/run.log`).
- The 1,000,000-token request limit is unchanged. `KV_CACHE_DTYPE=nvfp4_ds_mla` gives more
  capacity, but RP2 was never KL-measured with NVFP4 KV.

**Behaviour check.** The DCP4 harness also tried three behaviour profiles. Those calls exited at
once with status 2 and no output, because v0.4.29 has no `--reasoning-effort` option
(`window.log`). A behaviour check (LAVD, Estonia and Hotel Lights, 10 runs each) was run
separately. Its results are not in this repository yet and will be added.

### 7.4 GPU 3 at 350 W (2026-09-26)

After the A/Bs above, the owner raised GPU 3's power limit from 300 W to 350 W.
- GPU 3 is a Workstation card with a 600 W maximum.
- The Max-Q cards (GPUs 0 and 2) stay at their 300 W hardware maximum. GPU 1, the other
  Workstation card, stays at 300 W.

**Check.** Two passes on the production server, which runs the release configuration: RP2 image,
TP4/DCP4, MTP3, 48 sequences, 8,192 batched tokens, GMU 0.88, FP8 KV
(`results/speed-20260926/gpu3-350w/`, 2026-09-26, 14:15-14:30).
- **Order.** The 350 W pass ran first, then a 300 W pass. Single runs.
- **Method.** C1 decode with the greedy fixed-input method of the DCP1 table (`bench_fixed.py`,
  temperature 0, 20 s cells, at most 8,192 tokens); prefill with `llm_decode_bench.py
  --prefill-only`, cold, 20 s per context. `quick8000.sh` drove both passes.
- **Conditions.** Both passes had the +150 MHz GPC offset on GPU 3. A scheduled weekly job that
  uses the server was paused, and nothing else used the server.

| cell | GPU 3 at 300 W | GPU 3 at 350 W | 350 W / 300 W |
|---|---|---|---|
| C1 decode, 0K | 190.2 (0.498) | 203.6 (0.498) | 1.071 |
| C1 decode, 8K | 217.6 (0.627) | 210.6 (0.500) | 0.968 |
| C1 decode, 32K | 204.7 (0.539) | 198.9 (0.475) | 0.972 |
| C1 decode, 128K | 207.6 (0.662) | 212.3 (0.586) | 1.023 |
| prefill, 8K | 8,491 | 9,163 | 1.079 |
| prefill, 32K | 8,879 | 9,426 | 1.062 |
| prefill, 128K | 8,722 | 9,294 | 1.066 |

Decode values are tokens/s with MTP acceptance in parentheses, to three decimals as in
`summary.json`.

**Reading.**
- **Clocks.** GPU 3's median SM clock under load went from 2,085 MHz to 2,625 MHz. The Max-Q
  cards stayed at about 2,220 MHz (2,212-2,242 MHz) and now set the pace. GPU 1 ran at 2,407 MHz
  in the 300 W pass and 2,280 MHz in the 350 W pass. "Under load" means polls in which all four
  GPUs were at 90% utilization or more (`clocks.csv`; the medians are in `summary.json`).
- **Prefill** was 6-8% faster at 350 W.
- **Decode is confounded by MTP acceptance.** At equal acceptance (0K, 0.498 in both passes) it
  was 7% faster. Dividing each rate by the expected tokens per step (1 + 3a) gives an approximate
  step rate that is 5-12% higher at 350 W in every cell (`steps_per_s_approx` in `summary.json`).
- **Scope.** Both passes ran the release configuration. This is a power-limit comparison, not an
  A/B of RP2 against the reference kernels.

## 8. Reproduction

### 8.1 Images

| Role | Identity |
|---|---|
| Serving image (public) | `verdictai/trellismx:glm53-flash-p8-r27-reference-20260909@sha256:ca6b80188dce154b91f49108b7d87792d2ba6328935afc71b44d1c0e6f6a1adf` |
| Its base image, per the published image record | `voipmonitor/vllm:jovian-judgement-community-20260906-r27@sha256:a298fe1cd207eaf97bd2ff2686716ed25b7009c09b36650eba732a4a7dc51512` |
| TrellisMX capture image: control and RP arms of all windows (local build, not published; the image of the published 09-09 measurement) | image id `sha256:0405a1c0dc128b51069798a5d00b346257bbd006c0deb7e6530a3d005d75de71` |
| TR3 capture image: TR3 arm of the final run (local build, not published; the image of the published 09-09 TR3 comparison) | `local/tr3-r10:cf32-process-cache-20260909`, image id `sha256:62e069faf47f2d42eae5f2c1677f8730a2f3f93d576301fe1bcc7e55f2fdb673` |
| RP2 release image (public; section 7), built from `deploy/Dockerfile` on the serving image | `verdictai/trellismx:glm53-flash-p8-r27-rp2-20260926@sha256:a0392e1c370eb933d87511928ea3aba6d5d63965cfe6eeaf9cc026a667e98ddc` |

Each arm's docker argv is in `results/kld-cf128/<arm>/launch.json`, and the image digest for each
arm is recorded there. The published 09-09 `comparison.json` files record both capture images and
their provenance. The release speed runs record theirs in the `launch.json` files under
`results/speed-20260926/` and in `dcp1/control-launch.json` there.

### 8.2 Inputs

- **TrellisMX r27 checkpoint** at the revision in section 9. Its `trellismx-manifest.json` is
  byte-identical to the one we ran, and the sidecar sha256 values in
  `results/testB/layer-003.json` and `layer-008.json` match the Hub files.
- **NVFP4 carrier** at the pinned revision.
- **TR3 4bpw** at `aba59d2175e1ee2887ae0ae1300ba848b1deed84`.
- **BF16 base model** at `a6c167b62691b2bac901344b65cb651a70f53e43`. Tests B and B2 need only the
  routed-expert shards of the tested layers.
- **Teacher dataset**, at two revisions:
  - `95f4fdd94bf29989db2e0d1054e4931f55edb6aa` holds the routed-block captures.
    `scripts/fetch_fit_capture.py` fetches only the fit windows, by byte range, and checks them
    against the capture manifest.
  - `7c378d5f17dba158c4c803eff27c346dd0615660` holds the conditional-fit teacher logits and token
    arrays.
- **Window lists.** `results/kld-rp-20260925/verified-inputs.json` (32 windows) and
  `results/kld-cf128/verified-inputs.json` (all 128, in manifest order) record, for every window:
  - the token-array and token-value sha256;
  - the teacher-logit file and its sha256;
  - the role manifest and metric hashes.
- **Campaign repository** at the cited commit, for the `glm53_nvfp4` modules.

Local paths in the scripts are placeholders (`<workspace>`, `<home>`, `<data-volume>`,
`<model-volume>`); see `scripts/README.md`.

### 8.3 Commands

1. **Kernel trees.** Apply `patches/*.diff` to the image's b12x, or run the builders:
   - `scripts/build_zero_remainder_b12x.py`
   - `scripts/build_dx2_b12x.py`
   - `DX2_DST=<scratch dir> scripts/build_dx2_rp2_b12x.py`. The host guards are on by default;
     with `DX2_GUARDS=0` the output is byte-identical to the measured `b12x-dx2rp2` tree.
   - `EPIPAR_FILES=route_hoist_k5.py EPIPAR_DST=<scratch dir> scripts/build_rp2_epipar_b12x.py`
     for the release tree. The EPI-PAR check trees were built with `DX2_GUARDS=0`: without
     `EPIPAR_FILES` for the M1 and M2-16 build, and with it for the M1-only build.
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
7. **RP2 smoke and guards.** `scripts/run_rp2_smoke.sh` and `scripts/run_rp2_guard_smoke.sh`. The
   guard run needs a guarded tree, which is the builder's default.
8. **End-to-end windows.**
   - The window scripts are `scripts/window_nvfp4_20260925.py`, `scripts/window_fp8_20260925.py`
     and `scripts/window_cf128_20260925.py`.
   - The 128-window inputs come from `scripts/build_inputs_128.py`.
   - They need the 09-09 reference launch arguments, the capture images above and, for the final
     run, the TR3 attempt-2 launch.
   - They also operate our production host (systemd user unit, locks, ports). Treat them as a
     record of the procedure and adapt them before use.
9. **EPI-PAR check and release pre-flight.**
   - `B12X_TREE=<check tree> EP_OUT=/work/results/<name>.json scripts/run_epipar_check.sh`.
   - `MODE=save scripts/run_final_identity.sh`, then `MODE=compare scripts/run_final_identity.sh`
     with the release image.
10. **Release speed.** The harnesses are next to their results: `results/speed-20260926/dcp1/`
    (`run.py`, `quick.sh`, `bench_fixed.py`), `results/speed-20260926/dcp4/` (`window.py`) and
    `results/speed-20260926/gpu3-350w/` (`quick8000.sh`). Like the window scripts, they operate
    our host and are a record of the procedure.

### 8.4 Re-deriving the reported numbers (no GPU)

- **Timing and layer-level analyses.**
  - `analyze_timing.py`, `analyze_timing_split.py`, `analyze_BC.py` and `analyze_B2_A2.py`
    regenerate `timing_analysis.json`, `timing_split_analysis.json`, `analysis_BC.json` and
    `analysis_amendment2.json` exactly from the raw files in `results/`.
  - `summarize_rowpack_timing.py` reproduces the A3 and A4 medians and intervals.
- **Final report.** `python3 scripts/final_cf128_report.py` runs from this layout. It reproduces
  every field of `results/kld-cf128/final-report.json` except the KV capacities.
  - It reads those from engine startup logs that are not included. The values are in the runtime
    audits and in `results/kv_capacity_serving_profile.json`.
  - It also rewrites that file, so run it on a copy.
- **Extras.** `python3 scripts/reported_extras_cf128.py` runs from this layout. It reproduces every
  field of `results/kld-cf128/reported-extras.json`, including the Bonferroni and ratio-of-means
  intervals (`primary_robustness`), except `window_dependence`, which needs the local token
  arrays.
  - It also rewrites that file, so run it on a copy.
  - The published file is the workspace run, which includes `window_dependence`.
- **Release speed.** `python3 results/speed-20260926/dcp4/summarize.py` regenerates
  `summary.json` byte for byte from the raw files next to it; run it on a copy. The DCP1 values
  are `aggregate_tps` and `server_spec_accept_rate` of each `decode-*-c1.json`, and
  `tok_per_sec` of each `prefill-quick.json`. The GPU 3 check reads the same fields from
  `gpu3-350w/w300/` and `gpu3-350w/w350/`. Its clock medians follow from `gpu3-350w/clocks.csv`
  (columns: time, GPU, SM clock, power, utilization, temperature): take each four-GPU poll with
  every GPU at 90% utilization or more, split at the gap between the passes (14:21:57-14:22:51),
  and take the median per GPU.
- **KL divergence.** Every production window includes three kinds of score record:
  - `scores/*.json`, the per-window records. Each window's `true_decode_mean_kld` is the mean of
    `kld[1:]` in the matching `.npz`.
  - `scores/*.npz`, the per-row scores: row KL, teacher entropy, top-1 tokens and probabilities,
    and realized-token log-probabilities.
  - `scores/*.retirement.json`, hash records tying each score file to its retired raw logits.
- **Intervals.** The intervals in each `analysis.json` are `scipy.stats.bootstrap` BCa over the
  window-paired values. The cross-run comparisons use the same estimator against the 09-09
  `comparison.json` files, which are included in `results/reference-20260909/`.

**Not included:** `.pt` tensors, including the pre-flight's saved outputs; raw logits, which were
retired after hashing and scoring; request/response captures; server and startup logs;
telemetry and clock logs, except `results/speed-20260926/gpu3-350w/clocks.csv`.

## 9. Sources

### 9.1 Models, data, code and images used

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
   (https://huggingface.co/brandonmusic/GLM-5.3-Flash-TrellisMX-MXFP8).
   - It is the checkpoint under test.
   - That revision holds the published 09-09 KL results we compare against, under
     `results/kld-reference-20260909/` and `results/kld-tr3-20260909/`.
   - Byte-identical copies of those files (`comparison.json`, `audit.json`, `README.md`) are in
     this repository under `results/reference-20260909/`.
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
     - `p8_decode_protocol`, the KL capture and scoring protocol. It was imported from a local
       checkout of the same repository and is byte-identical to the file at this commit.
   - Its `results/P8_COUPLED_INCOHERENCE_ARCHIVE_AUDIT.md` holds the published 09-04 screen value
     that Test B reproduces.
7. **Serving image.** `verdictai/trellismx:glm53-flash-p8-r27-reference-20260909` (digest in
   section 8.1). Every kernel test and every TrellisMX production arm ran in it or in its capture
   derivative, except the compare step of the release pre-flight, which ran in the release image.
   The TR3 arm used its own capture image (section 8.1). The serving image is also the base of the
   RP2 release image and the reference arm of the release speed A/Bs (section 7).
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
   number. The DCP1 release decode cells ran it through the fixed-input wrapper
   `results/speed-20260926/dcp1/bench_fixed.py`, which checks that sha256 before it runs.

### 9.2 Background

- **QTIP.** Tseng, A., Sun, Q., Hou, D., and De Sa, C. "QTIP: Quantization with Trellises and
  Incoherence Processing." NeurIPS 2024. arXiv:2406.11235. Cited as the origin of the trellis
  codes with computed codebooks that the P8 weight format uses; the b12x intrinsics label the
  trellis path "QTIP/EXL3".
- **ExLlamaV3 / EXL3.** turboderp. https://github.com/turboderp-org/exllamav3. Cited because the P8
  decode's procedural MCG constants and state construction are ported from ExLlamaV3's MCG decoder,
  as the b12x source headers state.
- **Microscaling.** Open Compute Project, "OCP Microscaling Formats (MX) Specification v1.0," 2023;
  Rouhani, B. D., et al. "Microscaling Data Formats for Deep Learning." arXiv:2310.10537, 2023.
  Cited because they define the E4M3 elements with UE8M0 scales per 32-element block that the P8
  activations and the remainder use.
- **NVIDIA PTX ISA.** https://docs.nvidia.com/cuda/parallel-thread-execution/. Cited for the
  block-scaled `mma.sync` (`kind::mxf8f6f4`) that the kernels issue, and for the NVFP4 format
  (E2M1 values with E4M3 scales) of the KV record.
- **QuaRot.** Ashkboos, S., Mohtashami, A., Croci, M., Li, B., Jaggi, M., Alistarh, D., Hoefler,
  T., and Hensman, J. "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs." NeurIPS 2024.
  arXiv:2404.00456. Cited as the rotation-before-quantization idea that Test C tested on the KV
  record.
- **KL divergence.** Kullback, S., and Leibler, R. A. "On Information and Sufficiency." Annals of
  Mathematical Statistics 22(1), 1951. Cited because it defines the primary metric.
- **BCa intervals.** Efron, B. "Better Bootstrap Confidence Intervals." Journal of the American
  Statistical Association 82(397), 1987. Cited because the preregistered KL intervals are BCa
  intervals. The sensitivity analyses use percentile bootstraps where stated (section 3.13).
- **SciPy.** `scipy.stats.bootstrap` (`method="BCa"`). Cited because it computes the KL intervals.
- **Preregistration.** Nosek, B. A., Ebersole, C. R., DeHaven, A. C., and Mellor, D. T. "The
  preregistration revolution." PNAS 115(11), 2018. Cited for the practice this study follows: a
  plan and decision rules fixed before the data, with amendments stated openly.

## 10. License

This repository's own scripts, documentation and result files are distributed under the ShapleyMCG
License 1.0 (`LICENSE`), the same license as the campaign repository. It is source-available, not
OSI open source. Required attribution:

> ShapleyMCG was created by Brandon M. Music. Canonical source:
> https://github.com/brandonmmusic-max/shapleymcg

The b12x kernel patches in `patches/` modify Apache-2.0 b12x sources and are provided under the
Apache License 2.0 (`patches/LICENSE.b12x`). The files in `results/reference-20260909/` are copies
of files published with the TrellisMX model. The files in `deploy/` are this repository's own and
fall under the ShapleyMCG License; the image they describe contains third-party software under its
own licenses. Third-party models, datasets, images and libraries keep their own licenses.

## 11. Repository layout

```
README.md                 this document
PREREG.md                 preregistration, Amendments 1-7 and the commit chronology
RESULTS.md                results log, written as results came in, with the audit corrections
LICENSE                   ShapleyMCG License 1.0
scripts/                  every script that produced a number (see scripts/README.md)
patches/                  b12x kernel diffs and their Apache-2.0 license
deploy/                   RP2 release image: compose.yaml, serve-rp2.sh, Dockerfile, image record
results/
  timing_raw.json, timing_analysis.json                Test A
  timing_split_raw.json, timing_split_analysis.json    Test A exploratory split
  gpucheck_*.json                                      per-GPU M1 timing and clock checks
  testB/, testB2/                                      layer-level records, 09-04 screen reproduction
  testC/                                               Test C records and the KV-mirror check
  analysis_BC.json, analysis_amendment2.json           Tests B, C, B2 and A2 analysis
  k2/closure-*.json                                    K2, K3 and K4 closure
  a2_timing_raw.json, a3_timing_raw.json               A2 and A3 timing
  a4_timing_raw.json, a4_timing_summary.json           A4 timing
  dx2_split_timing_raw.json                            A2 exploratory split
  rp2_smoke_20260925_console.jsonl                     RP2 smoke output, 09-25
  rp2_guard_smoke.json, rp2_guard_smoke.log            RP2 smoke rerun and guard test, 09-26
  kv_capacity_serving_profile.json                     serving-profile KV capacities
  reference-20260909/                                  copies of the published 09-09 KL comparison files
  kld-rp-20260925/                                     first production window (NVFP4 KV, 32 windows)
  kld-fp8-20260925/                                    second production window (FP8 KV, 32 windows)
  kld-cf128/                                           final run (FP8 KV, 128 windows):
    analysis.json, final-report.json                     preregistered analysis and final report
    reported-extras.json                                 reported-only and audit sensitivity analyses
    verified-inputs.json                                 window list with input and teacher hashes
    control-fp8/, rp2-fp8/, tr3-fp8/                     launch argv, runtime audits, per-window scores
    speed/rp2-fp8/                                       benchmark outputs and launch arguments
    window.log, execution*.json, restoration.json        window log, execution and restoration records
  epipar_check_*.json, epipar_check_*.log              EPI-PAR identity and timing (section 4.10)
  final_identity_check.json, final_identity_*.log      release pre-flight (section 7.2)
  speed-20260926/                                      release speed A/Bs (section 7.3):
    speed-summary-20260926.json                          both tables and their conditions
    dcp1/                                                DCP1 A/B: outputs, launch argv, harness
    dcp4/                                                DCP4 A/B: outputs, audits, summary, harness
    gpu3-350w/                                           GPU 3 at 300 W and 350 W (section 7.4)
```

`RESULTS.md` and `PREREG.md` name files by their paths in the testing workspace:
- `kld-*/...` there is `results/kld-*/...` here;
- `results/b12x-dx2-guards.diff` is `patches/b12x-dx2-guards.diff`.

The release speed harnesses name the workspace directories: `dcp1-speed-20260926/` there is
`results/speed-20260926/dcp1/` here, `bench-20260926/` is `results/speed-20260926/dcp4/`, and
`prod-speed-gpu3-20260926/` is `results/speed-20260926/gpu3-350w/`.
