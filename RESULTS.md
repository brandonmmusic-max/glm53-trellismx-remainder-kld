# Results: FP8 remainder prologue + MLA-latent Hadamard (TrellisMX r27), 2026-09-23

Production (`trellismx-reference-production-20260909`) served throughout. Nothing in
production was changed. Only roles-v3 `fit` windows were read. See PREREG.md, including
Amendment 1.

## Test A: zero-remainder timing (preregistered). **FAIL**

A second `mxfp8` MMA was added per decoded weight fragment (A operand really loaded and then
zeroed at runtime). Every output was bit-identical to baseline (`torch.equal`), baseline was
deterministic, and the unmasked positive control changed the outputs.

Setup: CUDA-graph replay, 60 interleaved blocks x 200 replays x 4 input sets per cell, on
GPU 0.

| layer | M1 | M4 | M16 (reported only) |
|---|---|---|---|
| 8 (K4) | 1.0698 [1.0697, 1.0699] | 1.1650 [1.1648, 1.1653] | 1.1341 |
| 3 (K5) | 1.0854 [1.0853, 1.0855] | 1.1484 [1.1406, 1.1553] | 1.1155 |

The bar was <= 1.03 at M1 and M4. Per the PREREG, the two-hop remainder stops here.
(`results/timing_analysis.json`)

### Exploratory (after the FAIL, not preregistered): where the cost is

The same harness ran on GPU 2, splitting the extra MMA by GEMM. All arms were exact, and the
FC2 positive control changed the outputs.

| layer | M | both | FC2 (down) only | FC1 (gate/up) only |
|---|---|---|---|---|
| 8 K4 | 1 | 1.0786 | **1.0244** | 1.0779 |
| 8 K4 | 4 | 1.1741 | **1.0247** | 1.1675 |
| 8 K4 | 16 | 1.1498 | **1.0229** | 1.1296 |
| 3 K5 | 1 | 1.0758 | **1.0009** | 1.0748 |
| 3 K5 | 4 | 1.1532 | **1.0240** | 1.1228 |
| 3 K5 | 16 | 1.1189 | **1.0186** | 1.0980 |

The ZR arm is a lower bound on a real remainder's cost. It does not include:
- the lo-term quantize in the FC1 epilogue;
- the lo-scale loads;
- staging the lo tile.

Only the decode kernels (M <= 16) were timed. (`results/timing_split_analysis.json`)

## Test B: remainder numerics, layer level (preregistered). Criteria met

Harness: the campaign's `p8_layer_rate_damage` method.
- r27 rank sidecars, reference-decoded; the coupled reference forward; damage against the
  BF16 source experts.
- 64 fit windows x 48 tokens.
- The production arm is asserted bit-identical to `coupled_expert_reference`.
- The BF16-closure arm lands at 4-9e-8 relative.

Like-for-like check: the 09-04 screen's ordinary-E4M3 geometric-mean NMSE reproduces as
7.3725278e-4 vs the published 7.3725270e-4 (relative difference 1e-7). On the same frozen
plan, the two-term carrier gives 2.76e-7 (16/16 experts improve).

- **Pooled D(P-A8x2)/D(P-A8) = 0.490 (-51.0%), BCa 95% CI [0.427, 0.562], 6/6 layers improve.**

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

Per-domain two-term ratio: general 0.596, legal 0.509, code/agentic 0.625, reasoning 0.403.

This is a layer-level proxy, not KLD. Under the PREREG, the Test A failure stops the
two-hop design regardless of this result. (`results/analysis_BC.json`)

## Test C: fixed H512 before the NVFP4 MLA latent record (preregistered, amended). **FAIL**

The torch mirror of the production writer was checked against the real
`concat_and_cache_glm_next_mla` (304-byte two-level record) on real stand-in latents:
99.96% of elements are identical, and error energy agrees to 1.6e-5.

Decisive layers are the MLA layers 7, 15, 19, 27, 35 and 43, with stand-ins from captures
6, 14, 18, 26, 34 and 42.

| metric | H512 pooled ratio | H512 x random signs |
|---|---|---|
| latent NMSE (U = 0.0090) | 1.0011 [1.0009, 1.0013] | 1.0006 [1.0004, 1.0008] |
| attention-logit error | 1.0055 [1.0044, 1.0066] | 1.0056 [1.0044, 1.0068] |

Why it fails: the latent after `kv_a_layernorm` is already Gaussian-shaped.
- Per-token amax/RMS is 3.25, the value for a 512-dim Gaussian.
- The NVFP4 error of 0.0090 NMSE is the Gaussian floor.
- There are no outlier channels for a rotation to spread.

The same-layer stand-ins (3:3, 23:23, 43:43) agree within +/-1.6%.

## Side finding: GPU 0 is pinned at 1642 MHz

The same M1 MoE graph was timed one GPU at a time while production was idle. All four GPUs
report NVML mem offset +6000, gpc offset 0, identical P0 ranges, and no throttle reasons.

| GPU | M1 time | clock under load | power |
|---|---|---|---|
| 0 (Max-Q) | 108.5 us | flat 1642 MHz | 91 W |
| 2 (Max-Q) | 77.8 us | flat 2295 MHz | 153 W |
| 1 | 63.5 us | 2865-2872 MHz | 261 W |
| 3 | 63.5 us | 2820-2842 MHz | 257 W |

This is the 09-13 stale `-lgc` signature. No boot service locks clocks. Not changed.
(`results/gpucheck_*.json`, `logs/clk_gpu*.csv`)

## GPU 0 unpinned (Brandon's OK, 09-23)

`sudo -n nvidia-smi -i 0 -rgc`: the M1 kernel went from 108.5 us to 77.8 us, the same as its
twin GPU 2. It then ran at 2167-2295 MHz / 261 W. The +6000 offsets and 300 W limits were
unchanged.

## Amendment 2: down-hop remainder, plane-based D-x2 (preregistered)

**K2, real-kernel closure: PASS.** Layers 3 and 8, all four ranks summed, 3,072 fit tokens,
every path (M1, M16, M64, M3072).

| layer | path | D_kernel(prod)/D_ref(P-A8) | D-x2/prod (kernel) | reference down-only |
|---|---|---|---|---|
| 3 | M1 (96 tok) | 1.0025 | 0.3650 | 0.3630 |
| 3 | M16/M64/M3072 | 1.0015 | 0.6189 | 0.6183 |
| 8 | M1 (96 tok) | 1.0008 | 0.7844 | 0.7843 |
| 8 | M16/M64/M3072 | 1.0003 | 0.9384 | 0.9384 |

With the flag off, the D-x2 build's outputs are bit-identical to the image's own kernels
(layers 3 and 8, M16 and M3072, four ranks summed).

**A2, real-kernel timing: FAIL** (decode and prefill). D-x2/prod medians:

| layer | M1 | M4 | M16 | M64 | M512 | M2048 |
|---|---|---|---|---|---|---|
| 8 K4 | 1.026 | **1.079** | 1.009 | 1.053 | **1.172** | **1.187** |
| 3 K5 | **1.030** | **1.071** | 1.038 | 1.047 | **1.163** | **1.185** |

Exploratory split (not preregistered): the cost is mostly FC2 (lo staging plus the extra
MMA), e.g. +15-16% at M512. The FC1 epilogue adds 0-3%.

**B2, fresh-layer numerics: PASS.** Layers 6, 14, 26 (K4) and 21, 34, 42 (K5).

- **Pooled down-only D ratio: 0.744 (-25.6%), BCa 95% CI [0.673, 0.807], 6/6 improve.**
- Per layer, down-only / both-hops / activation share:

| layer | K | down-only | both-hops | activation share |
|---|---|---|---|---|
| 6 | 4 | 0.939 | 0.865 | 0.14 |
| 14 | 4 | 0.901 | 0.840 | 0.16 |
| 26 | 4 | 0.896 | 0.851 | 0.15 |
| 21 | 5 | 0.804 | 0.802 | 0.23 |
| 34 | 5 | 0.747 | 0.719 | 0.29 |
| 42 | 5 | 0.372 | 0.282 | 0.72 |

The fresh-layer gain is smaller than on the Test B layers (post-hoc down-only 0.606): the
K4 layers with high weight error gain little.

## Amendment 3: row-packed D-x2-RP (preregistered)

Design: on the direct decode paths every MMA tile holds one route row. The lo term goes into
row +8 of the route's own tile, and FC2 adds the row-8 accumulator into row 0 in register.
There is no extra MMA and no extra staging. Paths with M > 16 are unchanged.

**K3, closure: PASS.**
- M1/M16 damage ratios vs reference: layer 3 at 0.3650 / 0.6189 (ref 0.3630 / 0.6183);
  layer 8 at 0.7844 / 0.9384 (ref 0.7843 / 0.9384).
- Matches the plane-based D-x2 to FP32 rounding.
- M64 and M3072 outputs are bit-identical to prod.

**A3, timing: PASS.** RP/prod medians over 240 blocks:

| layer | M1 | M4 | M16 |
|---|---|---|---|
| 8 K4 | 1.0254 | 1.0200 | 0.9350 |
| 3 K5 | 1.0009 | 1.0272 [1.016, 1.039] | 1.0030 |

**Overall (Amendment 3 rule): K3, A3 and B2 all pass.** That justifies asking for a
production window for a decode-scored end-to-end KLD run. Prefill (M > 16) is untouched, so
the standard prefill-scored KLD cannot show the effect; the evaluation must push every token
through the M <= 16 path, e.g. 16-token scheduler chunks for both arms.

## Amendment 4: production window 2026-09-25 (19:00:43 to 20:00:59, production restored healthy)

**KLD** uses the 09-09 protocol unchanged: 32 CF windows, true decode, MTP off, NVFP4 KV,
one fresh server per arm. The RP server logged 168 `P8_DX2_ROWPACK_ACTIVE` lines
(42 layers x 4 ranks); the control server logged 0.

| arm | mean true-decode KLD | window BCa95 |
|---|---|---|
| control | 0.0350129 | [0.02911, 0.04317] |
| rp (D-x2-RP) | 0.0330631 | [0.02801, 0.03983] |

**rp - control = -0.00195 (-5.57%), paired BCa95 [-0.005975, +0.000118], rp lower in 19/32.**
Preregistered reading: **no detectable change** (the CI crosses 0 by 0.0001).

Post-hoc, reported only (65,472 rows):
- median row KL -1.5%;
- q90 / q99 / q99.9 at -5% / -5% / -8%;
- mean without the top 0.5% of rows (rows above the pooled q99.5 dropped from both arms) -4.0%;
- window medians lower in 23/32.

Server-run noise: control today vs the published 09-09 control (0.0354562) is -1.25% in the
mean and 3.2% per-window SD. KLD is heavy-tailed: the top 1% of rows carry 31% of it.

Paired against the published TR3 4bpw (NVFP4 KV, 0.03048):
- control +0.00453 [+0.00029, +0.01032];
- rp +0.00258 [-0.00119, +0.00577].

RP closes 43% of the gap. RP with NVFP4 KV is +3.5% vs TrellisMX 09-09 with FP8 KV (0.03195).

**Speed** (production config, MTP3, private port 8038, cooling gate passed before every cell,
single runs):

| cell | control | rp | MTP accept (control / rp) |
|---|---|---|---|
| C1 0K | 184.3 | 191.1 (+3.7%) | 0.436 / 0.553 |
| C4 0K | 300.1 | 301.2 (+0.4%) | 0.550 / 0.554 |
| C1 8K | 184.1 | 193.9 (+5.3%) | 0.538 / 0.610 |
| C4 8K | 300.7 | 299.0 (-0.6%) | 0.669 / 0.656 |

RP is not slower. The C1 gains track higher MTP acceptance in these runs. That could be a
benefit of more BF16-like target logits, or a difference in the generated text; single runs
cannot tell which.

## Amendment 5: FP8 MLA KV window, 2026-09-25 (20:29:49 to production restore)

Same protocol and windows as Amendment 4, with `KV_CACHE_DTYPE=fp8`. The control-fp8 server
logged 0 RP markers; the rp-fp8 server logged 168.

| arm | mean true-decode KLD | window BCa95 |
|---|---|---|
| control-fp8 | 0.0311958 | [0.02643, 0.03776] |
| rp-fp8 | 0.0309337 | [0.02617, 0.03750] |

**rp-fp8 - control-fp8 = -0.00026 (-0.84%), paired BCa95 [-0.001146, +0.001246], lower in
19/32: no detectable change.**

Paired comparisons (cross-run where noted, so they carry ~1-2% server-run noise):
- FP8 KV vs NVFP4 KV, control today: -10.9% [-0.00898, -0.00171], lower in 24/32
  (significant).
- FP8 KV vs NVFP4 KV, rp today: -6.4% [-0.00337, -0.00037], lower in 26/32.
- rp-fp8 vs today's production (control, NVFP4 KV): -11.7% [-0.00892, -0.00212], lower in
  26/32.
- rp-fp8 vs TR3 NVFP4 KV (09-09): +1.5% [-0.00315, +0.00357], lower in 15/32.
- rp-fp8 vs TR3 FP8 KV (09-09): +9.7% [-0.00132, +0.00640], lower in 13/32.
- control-fp8 vs the published 09-09 TrellisMX FP8 control (0.0319452): -2.3% in the mean;
  the largest per-window difference is 0.014.

Reading:
- FP8 KV is a real, significant improvement (about -11%), matching 09-09 (-0.0035).
- The row-packed down-hop remainder shows -5.6% under NVFP4 KV but -0.8% under FP8 KV. Both
  intervals cross 0, so its end-to-end effect is small (plausibly 0-6%) and not established
  on 32 windows.

FP8-KV speed (rp-fp8, same production config, cooling gate and single-run caveats as
Amendment 4), shown next to the NVFP4-KV runs from the earlier window:

| cell | control NVFP4 | rp NVFP4 | rp FP8 | rp FP8 MTP accept |
|---|---|---|---|---|
| C0 / conc 1 | 184.3 | 191.1 | 193.0 | 0.528 |
| C0 / conc 4 | 300.1 | 301.2 | 293.8 | 0.589 |
| C8192 / conc 1 | 184.1 | 193.9 | 193.4 | 0.487 |
| C8192 / conc 4 | 300.7 | 299.0 | 304.0 | 0.613 |

KV capacity (engine log):
- Production profile (GMU 0.88, 48 seqs): NVFP4 12,581,699 tokens; FP8 8,127,659 tokens
  (-35%).
- Capture profile (GMU 0.97, 1 seq): NVFP4 31,565,217 tokens; FP8 19,362,962 tokens.

Production was restored healthy at 21:17:30 (down 20:29:49 to 21:17:30).
