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

## Amendment 6: both-hop row-pack D-x2-RP2 (preregistered gates)

The smoke test on layer 8, rank 0, passed:
- deterministic and finite;
- M1/M4/M16: RP2 differs from prod by 3.7-3.9% and from RP by 2.7-2.8%;
- M64/M512: bit-identical to prod.

**K4 closure: PASS.** Four ranks summed, Test B fit tokens.

| layer | path | D_kernel(prod)/D_ref(P-A8) | RP2/prod (kernel) | reference both-hops P-A8x2/P-A8 |
|---|---|---|---|---|
| 3 | M1 (96 tok) | 1.0025 | 0.2531 | 0.2514 |
| 3 | M16 | 1.0015 | 0.5494 | 0.5486 |
| 8 | M1 (96 tok) | 1.0008 | 0.5154 | 0.5150 |
| 8 | M16 | 1.0003 | 0.8689 | 0.8689 |

M64 and M3072 are bit-identical to prod on both layers.

**A4 timing: PASS.** RP2/prod medians over 240 blocks:

| layer | M1 | M4 | M16 |
|---|---|---|---|
| 8 K4 | 1.0263 | 1.0064 [1.0009, 1.0128] | 0.9405 |
| 3 K5 | 1.0273 | 0.9879 [0.9843, 0.9917] | 1.0024 |

The input-hop row-pack adds essentially no cost on top of the down-hop RP. Layer-level
both-hop damage on the fresh B2 layers (reference `P-A8x2`) has a geometric mean of about
0.68 (-32%), against 0.744 for down-only.

## Amendment 7: final 128-window run, 2026-09-25/26 (FP8 MLA KV; preregistered primaries)

Production was stopped at 21:25:35 via `systemctl --user stop trellismx-klc.service`. The arms ran in the fixed
order:

| arm | server healthy | capture complete | activation proof |
|---|---|---|---|
| control-fp8 | 21:27:51 | 22:22:59 | 0 `P8_DX2_ROWPACK_ACTIVE` lines |
| rp2-fp8 | 22:26:00 | 23:22:17 | 168 `P8_DX2_ROWPACK_ACTIVE ... input_hop=1` lines |
| tr3-fp8 | 23:25:18 | 00:24:23 | 0 lines (TR3 stack) |

All 128 conditional-fit windows were scored in every arm (2,046 scored rows per window).

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

**Reading (preregistered rules).**
- **Primary 1: RP2 improves on the control.** KLD falls by 4.9%, the CI excludes 0, and RP2 is lower in
  105 of 128 windows.
- **Primary 2: RP2 is NOT tied with TR3.** RP2 is 8.3% above TR3 and the CI [+2.0%, +14.5%] excludes 0. It is not
  equivalent at +/-5% or at +/-10%.
- RP2 closes 40% of the control's gap to TR3 (0.004183 -> 0.002504).
- The interim look at 00:11 (47 windows: +5.4%, CI [-5.3%, +14.2%]) could not tell RP2 and TR3 apart. The full set
  can: on the 96 windows scored for the first time, RP2 - TR3 is +9.8% [+2.1%, +16.9%] (reported only).

**Reported only.**
- The 96 new windows alone: rp2 - control -4.1% [-6.6%, -0.7%] (lower in 76/96); control - tr3 +14.5% [+6.7%, +21.9%].
- The original 32 windows: rp2 - control -7.5% [-12.8%, +2.1%] (29/32 lower); rp2 - tr3 +3.4% [-9.6%, +13.7%].
- Run-to-run stability on the original 32 windows:
  - control-fp8 here vs window 2: 0.031174 vs 0.031196 (-0.07%);
  - tr3-fp8 here vs 09-09: 0.027889 vs 0.028190 (-1.07%).

  Run-level drift is about 1% or less, well below the 8.3% RP2-TR3 gap.
- Row-level ratios (all 128 x 2,046 rows):

  | comparison | mean | median | q90 | q99 | q99.9 | pooled-q99.5 trimmed | rows lower | window medians lower |
  |---|---|---|---|---|---|---|---|---|
  | rp2 vs control | -4.9% | -4.1% | -5.2% | -5.5% | -3.9% | -5.1% | 52.9% | 100/128 |
  | rp2 vs tr3 | +8.3% | +12.3% | +9.4% | +6.6% | +9.2% | +8.1% | 46.6% | 33/128 |
  | control vs tr3 | +13.9% | +17.1% | +15.5% | +12.8% | +13.7% | +13.5% | 44.9% | 32/128 |

- Per-domain means:

  | domain (windows) | control-fp8 | rp2-fp8 | tr3-fp8 |
  |---|---|---|---|
  | general (38) | 0.03624 | 0.03500 | 0.03113 |
  | legal (37) | 0.04187 | 0.03868 | 0.03160 |
  | code / agentic (37) | 0.03038 | 0.02933 | 0.03064 |
  | reasoning / termination (16) | 0.02111 | 0.02044 | 0.02293 |

  Per-domain paired differences (reported only, not preregistered). These are 12 intervals with no
  multiplicity correction, and each domain has 16-38 windows.

  | domain | rp2 - control | rp2 - tr3 | control - tr3 |
  |---|---|---|---|
  | general | -3.4% [-9.2, +4.7], 31/38 lower | +12.4% [+2.5, +22.7], 10/38 | +16.4% [+5.5, +28.5], 9/38 |
  | legal | -7.6% [-10.1, -6.0], 35/37 lower | +22.4% [+13.8, +27.6], 4/37 | +32.5% [+24.1, +38.1], 2/37 |
  | code / agentic | -3.5% [-7.0, +5.1], 30/37 lower | -4.3% [-19.0, +9.4], 13/37 | -0.9% [-15.6, +13.2], 14/37 |
  | reasoning / termination | -3.2% [-10.0, +5.0], 9/16 lower | -10.8% [-31.1, +9.5], 9/16 | -7.9% [-30.9, +7.4], 9/16 |

  The TR3 advantage sits in legal and general text. On code and reasoning, RP2 and TR3 cannot be told apart. RP2's
  largest gain over the control is on legal text.
- KV capacity in the capture profile (engine startup log): TrellisMX FP8 19,333,333 tokens; TR3 FP8 24,950,413 tokens.

**Scope of the result.**
- True decode with MTP0 and one sequence means every scored token ran the M1 direct path, where RP2 is active.
  Every position, including the KV entries it writes, is computed by decode steps.
- In serving, the prompt is prefilled by the unchanged prefill kernels. The prompt's hidden states and KV entries
  therefore carry control numerics, and only generated tokens use RP2. The served-model effect is expected to be
  smaller than the true-decode effect measured here. This was not measured.
- In serving, RP2 covers the direct kernels only (M <= 16 tokens per step). Examples:
  - with MTP3, one sequence verifies 4 tokens per step and 4 sequences verify 16, so both stay on the direct path;
  - 5 or more concurrent sequences with MTP3 (20+ tokens per step), and all prompt prefill, use the grouped and
    prefill kernels, which are unchanged (`p8_native_kernel.py`: `small_m = m <= 16` on the full-coupled path).
- FP8 MLA KV throughout. NVFP4-KV production numbers are in Amendments 4-5.

**Speed (descriptive, preregistered as such).**
- Setup: the rp2-fp8 production-config server on :8038 (MTP3, TP4/DCP4, FP8 MLA KV), started 00:24:23.
- Each cell is one run of `llm_decode_bench.py --skip-prefill --duration 30`, with the cooling gate before it
  (>= 90 s idle, all GPUs <= 55 C for 30 s).
- tok/s (MTP acceptance):

| cell | control NVFP4 (w1) | rp NVFP4 (w1) | rp FP8 (w2) | rp2 FP8 (w3) |
|---|---|---|---|---|
| 0K, C1 | 184.3 (0.436) | 191.1 (0.553) | 193.0 (0.528) | 196.4 (0.446) |
| 0K, C4 | 300.1 (0.550) | 301.2 (0.554) | 293.8 (0.589) | 297.9 (0.503) |
| 8K, C1 | 184.1 (0.538) | 193.9 (0.610) | 193.4 (0.487) | 200.7 (0.531) |
| 8K, C4 | 300.7 (0.669) | 299.0 (0.656) | 304.0 (0.613) | 312.7 (0.608) |

- These are single unpaired runs from three different windows, with no intervals. MTP acceptance varies between runs
  and moves C1 speed by several percent.
- The table shows no end-to-end slowdown from RP2. It cannot rank the arms.
- KV capacity in this serving profile with FP8 KV is 8,127,659 tokens, against 12.58M with NVFP4 KV (window 2).

**Close.** Production was restored via `systemctl --user start trellismx-klc.service`. It reported healthy at
00:42:03, with `/health` 200 and `/v1/models` serving `glm53-flash-trellismx-p8-k45`. Downtime was 21:25:35 to 00:42:03.
