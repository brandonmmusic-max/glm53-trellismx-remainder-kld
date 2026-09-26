# PREREG: FP8 remainder prologue + MLA-latent Hadamard for NVFP4 KV (TrellisMX r27)

Written 2026-09-23 before any number for these three tests existed. Production
(`trellismx-reference-production-20260909`) keeps serving throughout. Nothing here changes
production; these are small-scale gates for deciding whether an end-to-end KLD run is worth
a production window.

Protected roles: only roles-v3 `fit` windows are downloaded or read. Selection,
confirmation and final windows stay unopened.

## Test A: zero-remainder timing (is a second FP8 MMA free at decode?)

- **Kernel:** the arm production runs for the full-coupled r27 checkpoint at M <= 16
  (direct-route small-M, materialized), compiled from a scratch copy of the image's b12x.
- **Arm Z:** identical, except each decoded weight fragment feeds a second `mxf8f6f4` MMA
  whose A operand is zero, accumulating into the same FP32 accumulator. Outputs must equal
  the baseline exactly (torch.equal).
- **Payloads:** layer 3 (K5) and layer 8 (K4), TP rank 0 sidecars from
  `trellismx-r27-local-checkpoint-v1`; real routes (top-8 ids) and routed-block inputs from
  the fit capture; M in {1, 4, 16}.
- **Timing:** CUDA-graph replay, interleaved baseline/Z blocks, >= 30 blocks x 50 replays
  per arm per M. Metric: paired per-block time ratio Z/baseline, median with a bootstrap
  95% CI over blocks.
- **Decision:** PASS if the median ratio is <= 1.03 at M1 and M4 for both layers and every
  output check is exact. Otherwise the remainder direction stops. M16 is reported only.

## Test B: remainder prologue numerics (layer level)

- **Harness:** the campaign's `p8_layer_rate_damage` method, unchanged except for new
  activation arms:
  - r27 sidecars reference-decoded with the kernel's procedural-MCG E4M3 codebook;
  - the exact coupled reference forward (`coupled_expert_reference`);
  - routed-output damage `D = sum_t ||S(t)||^2 / sum_t ||routed_output(t)||^2` against the
    BF16 source experts.

  Coupled scales are reassembled from the r27 sidecars (contiguous-atom32 TP slices).
- **Layers:** 3, 33, 43 (K5) and 8, 18, 23 (K4). All 288 experts.
- **Tokens:** the 64 roles-v3 fit windows x 48 tokens (linspace), i.e. 3,072 tokens per
  layer.
- **Arms:** same tokens, same decoded weights.
  - `P-A8`: P8 weights, single-term E4M3/UE8M0-K32 at both hops (= production).
  - `P-A8x2`: P8 weights, two-term at both hops. `hi = q(x)`, `lo = q(x - hi)`; the kernel
    would issue two MMAs; the reference uses `hi + lo` in FP32.
  - `P-Aexact`: P8 weights, unquantized carriers (weight-only error).
  - `B-A8`, `B-A8x2`: BF16 weights mapped exactly into coupled coordinates, with single-
    or two-term activations (activation-only error).
- **Primary:** per-layer ratio `D(P-A8x2) / D(P-A8)`; pooled geometric mean over the 6
  layers.
- **Uncertainty:** paired window bootstrap of the per-layer log ratio (64 windows, BCa,
  20,000 replicates, seed 20260923).
- **Decision** ("worth an end-to-end KLD run"): all three must hold.
  - Pooled geometric-mean damage reduction >= 5%.
  - >= 5 of 6 layers improve.
  - The pooled log-ratio CI excludes zero.
- **Also reported, cannot change the decision:**
  - activation share `D(B-A8) / D(P-A8)` and weight share `D(P-Aexact) / D(P-A8)`;
  - K4 vs K5 split;
  - per-domain damage.
- **Like-for-like check:** rerun the 09-04 activation screen's `ordinary-e4m3` arm with its
  frozen plan (layer 3, its 16 experts, fit, domain-balanced, offset 32, count 32). The
  harness must reproduce its geometric-mean NMSE 7.3725e-4 within 1%; the same plan also
  gets a two-term arm.

## Test C: MLA-latent Hadamard before NVFP4 KV quantization

- **Record mirrored:** production's GLM5Next `nvfp4_ds_mla` latent record, 512-dim, no
  RoPE payload.
  - Per-token second-level scale `s_t = amax_t / (6 * 448)`.
  - Group-16 E4M3 scale bytes relative to `s_t`.
  - E2M1 values with RNE satfinite.
- **Arms:** unrotated latent vs normalized Sylvester H512 applied before quantization and
  inverted after dequantization. The rotation is orthogonal, so errors are compared in the
  same space; in serving, the inverse would fold into the absorbed W_UK / W_UV.
- **Latents:** approximate. Attention inputs are not in the capture, so the stand-in for
  layer L+1's attention input is `gamma_in(L+1) * (m_L / gamma_post(L))`. Here `m_L` is
  the captured routed-block input on the same fit tokens, and the latent is
  `kv_a_layernorm(kv_a_proj(.))` from the BF16 carrier. The same stand-in feeds the query
  path (`q_a_proj -> q_a_layernorm -> q_b_proj`, absorbed with each head's W_UK).
- **Metrics:**
  - (1) latent reconstruction NMSE (value path);
  - (2) relative attention-logit error `sum (q_abs . (c_hat - c))^2 / sum (q_abs . c)^2`
    over query/key pairs within each window.
- **Layers:** attention of layers 4, 9, 19, 24, 34, 44 (from captures 3, 8, 18, 23, 33, 43).
- **Decision** ("worth an end-to-end KLD run"): both metrics drop >= 20%, pooled over
  layers, with the window-bootstrap CI excluding zero.
- **Caveat:** stand-in latents. A pass is confirmed with real latents in the production
  window before anything ships.

## Amendment 1 (2026-09-23, written before any Test C number existed)

Reading the GLM5Next config and model code while writing Test C showed two facts the
original Test C text got wrong:

1. **Only every fourth layer is MLA.** `layer_types` puts DeepSeek sparse attention (the
   layers with the `nvfp4_ds_mla` latent record) at layers 3, 7, 11, ..., 43. Every other
   layer is KDA linear attention with a recurrent state and no latent cache. Of the
   preregistered layers 4, 9, 19, 24, 34, 44, only 19 is MLA.
2. **The residual is four hyper-connection streams** (`hc_mult` 4, Sinkhorn mixing). Each
   sublayer's input is a norm of a learned, token-dependent mix of the streams. So the
   stand-in `gamma_in(A) * (m_{A-1} / gamma_post(A-1))` is a normalized hidden state from the
   neighbouring sublayer, not a reconstruction of the true attention input. The caveat
   already required real-latent confirmation, and that requirement stands unchanged.

**Amended Test C:**

- **Layer set.** The same stand-in rule is kept (the capture of the sublayer immediately
  before attention A), but applied to MLA layers A in {7, 15, 19, 27, 35, 43}, using
  captures {6, 14, 18, 26, 34, 42} (fit windows only).
- **Tokens.**
  - Keys: all 2,048 tokens of each of the 64 fit windows.
  - Queries: the same 48 linspace tokens per window as Test B, over causal pairs.
- **Arms.**
  - Decisive: unrotated vs normalized Sylvester H512.
  - Reported only: H512 with fixed random signs (seed 20260923).
- **Record mirror.** The quantizer mirrors
  `b12x/attention/_shared/mla/kv_cache.py` (`per_token_scale=True`):
  - `s_t = token_amax/(6*448)`;
  - the scale byte is `e4m3_rn_satfinite(group_amax/s_t/6)`;
  - values are `e2m1_rne_satfinite(v/(dec(scale)*s_t))`.

  Exact reciprocals are used instead of `rcp.approx.ftz`.
- **Stand-in sensitivity (reported only).** For A in {3, 23, 43}, the latent is recomputed
  with the capture of layer A itself (the routed-block input right after attention A).
- **Decision rule and threshold.** Unchanged: both metrics drop >= 20% pooled (geometric mean
  over the six layers of the per-layer ratio), and the BCa window-bootstrap CI of the pooled
  log ratio (20,000 replicates, seed 20260923) excludes zero.

Test A's result was already known when this amendment was written (FAIL: +7.0% / +8.5% at
M1, +16.5% / +14.8% at M4). This amendment touches only Test C, which does not depend on it.
Test B's added `P-A8x2-fc1only` / `P-A8x2-downonly` arms were added after Test A and are
reported only.

## Amendment 2 (2026-09-23, written before any number for these tests existed): down-hop remainder confirmation

**Why.** Test A failed as preregistered, so the two-hop remainder stops. Two post-hoc results
pointed at the down (post-SwiGLU) hop alone:
- Test B's `P-A8x2-downonly` arm: pooled damage x0.606.
- The exploratory FC2-only zero-remainder timing: +0.1-2.5% of MoE-layer time.

Because both were found after the fact, they are confirmed here on fresh data and a real
kernel before anything moves toward production. GPU 0's stale clock pin was reset (with
Brandon's approval) before any of these runs.

**Design `D-x2`.**
- FC1 input: stays single-term E4M3/UE8M0-K32, as in production.
- FC1 epilogue: after quantizing each 32-block of the down input
  (`hi = q(v)`, same permuted K order), it also writes `lo = q(v - dq(hi))`, with its own
  UE8M0 scale, into a second plane of the intermediate buffer.
- FC2: stages both planes and issues a second `mxfp8` MMA per decoded weight fragment
  (`lo` A fragment, `lo` scale, same B and SFB), accumulating into the same FP32 accumulator.
- Kernels: every full-coupled path, i.e. M1, M2-16 direct, M17-128 grouped-M32 and
  M>128 prefill.

### B2: fresh-layer numerics (reference harness)

- **Layers:** 6, 14, 26 (K4) and 21, 34, 42 (K5). None was used in Test B.
- **Method:** `scripts/remainder_damage.py` unchanged, i.e. the same fit tokens (64 windows x 48)
  and the same BF16 source damage.
- **Primary:** pooled geometric-mean `D(P-A8x2-downonly)/D(P-A8)`, with the paired
  window BCa bootstrap from `analyze_BC.py` (20,000 replicates, seed 20260923).
- **PASS** requires all three: pooled reduction >= 20%; >= 5 of 6 layers improve; the CI upper
  bound of the pooled log ratio < 0.

### K2: real-kernel numerics closure

- **Setup:** layers 3 (K5) and 8 (K4); all four TP4 rank sidecars run one at a time on one GPU
  and their outputs are summed (what the all-reduce computes); the same 3,072 fit tokens per
  layer as Test B.
- **Coverage:** tokens are fed in chunks that exercise every path:
  - M1 (the first 64 tokens);
  - M16 chunks (all tokens);
  - M64 chunks (M17-128 path);
  - M3072 (prefill path).
- **Damage:** `D_kernel(arm) = sum_t ||kernel_t - base_t||^2`, with `base_t` the BF16
  source routed output, for arms `prod` (production kernels) and `D-x2`.
- **PASS**, per layer and per path:
  - `|D_kernel(prod)/D_ref(P-A8) - 1| <= 5%`, i.e. the harness and the kernels measure the
    same thing;
  - `D_kernel(D-x2)/D_kernel(prod)` within +/-0.05 of the reference
    `D(P-A8x2-downonly)/D(P-A8)` for that layer.
- The unmodified classes keep the new code behind `const_expr` flags, as in Test A. The
  production-arm outputs are also checked bit-identical to the image's own kernels.

### A2: real-kernel timing

- **Setup:** layers 3 and 8, rank 0, CUDA-graph replay, interleaved blocks with rotating order,
  >= 60 blocks x 200 replays per cell, on one production GPU with production idle, the same
  way as Test A.
- **Decode:** M in {1, 4, 16}. **PASS** if the median ratio `D-x2/prod` <= 1.03 at M1 and M4 on
  both layers.
- **Prefill:** M in {64, 512, 2048}. **PASS** if the median ratio <= 1.05 at M512 and M2048 on
  both layers.
- **Overall.** Asking for a production window for the end-to-end KLD run is justified only if
  B2, K2 and A2-decode all pass.
  - If A2-prefill fails while the rest pass, the only remaining option is a decode-only
    variant, which would need a decode-path KLD evaluation. It is reported, not assumed.

## Amendment 3 (2026-09-23, written before any number for D-x2-RP existed): row-packed down-hop remainder

**Outcome so far.** Amendment 2's A2 failed: the plane-based D-x2 cost +7-8% at M4 and
+16-19% in prefill. A post-hoc breakdown (not preregistered) put the decode cost mainly in
FC2's extra lo staging and extra MMA.

**Key fact.** On the direct decode paths (M <= 16) every physical 16/32-row MMA tile holds
exactly one route row (`valid_rows == 1`). Rows 1-15 of the A tile are zeros that the tensor
core already multiplies.

**Design `D-x2-RP`** (decode direct paths only):
- The FC1 epilogue writes `lo = q(v - dq(hi))`, with its own UE8M0 scale, into row +8 of the
  route's own tile (its payload words and its scale byte). This is the same math as D-x2.
- FC2 is unchanged except its epilogue. Row 8 is already staged and multiplied, with its
  A scale supplied by the odd lanes. The epilogue adds the row-8 accumulator into row 0 in
  register (`fragment[0] += fragment[2]`, `fragment[1] += fragment[3]`) before the existing
  FP16 store.
- There is no extra MMA, no extra staging and no second plane.
- M > 16 paths (grouped M32 and M64 prefill) keep the production kernels (single-term).
  Deployment would therefore be decode-only and would need a decode-scored KLD evaluation.
- Switch: `B12X_P8_DOWN_REMAINDER=rp` at construction. It joins the compile spec, and the
  phase kernels are upgraded to the static `...RP` classes only when `small_m`.

**K3: closure.**
- **Setup:** as K2, i.e. layers 3 and 8, all four ranks summed, the Test B fit tokens.
- **Paths:** M1 (first 96 tokens) and M16 (all 3,072).
- **PASS**, per layer and path:
  - `|D_kernel(prod)/D_ref(P-A8) - 1| <= 5%`;
  - `D_kernel(RP)/D_kernel(prod)` within +/-0.05 of the reference `P-A8x2-downonly` ratio;
  - RP outputs on the M64 and M3072 paths bit-identical to prod.

**A3: timing.**
- **Setup:** as A2, i.e. layers 3 and 8, rank 0, >= 60 interleaved blocks x 200 replays, M in
  {1, 4, 16}.
- **PASS** if the median ratio `RP/prod` <= 1.03 at M1 and M4 on both layers. M16 is reported.

**Numerics on fresh layers.** B2's down-only arm is the identical math, so B2's decision
stands for RP.

**Overall.** Asking Brandon for a production window (a decode-scored KLD run) is justified
only if K3, A3 and B2 all pass.

## Amendment 4 (2026-09-25, written before any end-to-end number): decode-scored KLD + speed, production window

Brandon approved the production window on 09-25 ("go ahead and do that, you can do it now").

**KLD. The 09-09 reference protocol is reused unchanged.** Source:
`trellismx-reference-release-20260909/kld/run.py`.
- **Windows and teacher:** the same 32 already-opened conditional-fit windows and BF16 teacher
  (HF dataset revision 7c378d5f, per-file sha256 verified).
- **Capture:** the capture-only image `trellismx-reference:cf32-20260909`
  (sha256:0405a1c0...), built from the production image ca6b8018. It runs forced-token true
  decode with 2048 inputs and 2047 captured rows.
- **Scoring:** row 0 is excluded, leaving 2046 true-decode rows per window, scored on CPU in
  FP64 over the 154,880-token vocabulary.
- **Server:** TP4/DCP4, MTP off, maxseq 1, NVFP4 MLA KV, batch 4096, GMU 0.97, maxlen 1M,
  prefix caching on with a zero-hit gate, and the same NCCL/PCIe settings.

**Arms**, one fresh server each, fixed order `control` then `rp`:
- `control`: exactly the 09-09 NVFP4 launch.
- `rp`: identical, plus the D-x2-RP b12x (`b12x-dx2`, built by `scripts/build_dx2_b12x.py`,
  byte-identical to the image's b12x except the gated D-x2 code) mounted read-only over
  `/opt/glm53-flash/b12x`, and `B12X_P8_DOWN_REMAINDER=rp`.
  - The server log must show `P8_DX2_ROWPACK_ACTIVE` for `rp` and must not show it for
    `control`.
  - `rp` uses a copy of the production JIT volume so production's cache is never written by
    the patched build.

**Primary:** the paired per-window difference `rp - control` in true-decode mean KLD, with a
BCa 95% interval (20,000 resamples, seed 20260902; the same estimator as the published
intervals).

**Reported:**
- each arm's mean KLD and window BCa interval;
- the number of windows where `rp` is lower;
- control reproduction against the published 09-09 NVFP4 value 0.0354562 (per-window
  comparison).

**Reading of the result:**
- **Improvement:** the paired mean is < 0 and the paired CI excludes 0.
- **No detectable change:** the CI includes 0.
- **Harm:** the mean is > 0 and the CI excludes 0.

This is a development measurement on opened windows, not final qualification. It is one server
preparation per arm, so the window intervals do not capture server-run variability.

**Speed** is descriptive, not a gate. The production serving config (MTP3, NVFP4 KV,
maxseq 48, batch 8192, GMU 0.88) runs on a private port 8038 for both arms (`control` first,
then `rp`, same JIT volumes as above).
- **Tool:** `llm_decode_bench.py` from `trellismx-performance-audit-20260908`.
- **Cells:** `--skip-prefill`, contexts 0 and 8192, concurrency 1 and 4, 30 s cells,
  2048 max tokens.
- **Cooling gate before each run:** >= 90 s idle, and all GPUs <= 55 C for 30 s.
- **Clock check:** GPU 0 vs GPU 2 clocks are checked under load (no pin).
- **Close:** production (`trellismx-reference-production-20260909`, unchanged) is restarted on
  :8000 and verified with `/health` and `/v1/models`.

## Amendment 5 (2026-09-25 ~20:10, before any number): FP8 MLA KV window, then an FC1-input row-pack

Brandon asked to try FP8 KV (#1), then the same row trick on the first input hop (#2), and
to remeasure.

**Window 2 (FP8 KV).** Same 09-09 KLD protocol, windows and scoring as Amendment 4, with
`KV_CACHE_DTYPE=fp8`.
- **Arms**, fixed order: `control-fp8` (production kernels) then `rp-fp8` (D-x2-RP, as
  Amendment 4).
- **Primary:** paired `rp-fp8 - control-fp8`, BCa95 (20,000 resamples, seed 20260902).
- **Reported** (cross-run, so they carry ~1.3% server-run noise):
  - `rp-fp8` vs TR3 FP8 KV (0.0281899);
  - `rp-fp8` vs today's `rp` NVFP4 KV (0.0330631);
  - `control-fp8` vs the 09-09 TrellisMX FP8 value (0.0319452);
  - KV capacity (tokens) from each server's startup log.
- **Speed:** only `rp-fp8`, same four cells and cooling gate as Amendment 4, compared
  descriptively with today's `rp` NVFP4 speed.
- **Close:** production is restored exactly as in Amendment 4.

**FC1-input row-pack (D-x2-RP2), prereg to be appended before its numbers.** The same
spare-row trick is applied to the input hop on the direct decode paths.
- Build `x_lo = q(x - dq(x_hi))`.
- FC1 feeds it as MMA rows q+8, instead of the broadcast duplicate, with the odd-lane A
  scales.
- The FC1 epilogue adds the row-8 accumulators into row 0 before the pair-scale / H128 /
  SwiGLU boundary.
- Gates as before: closure vs the reference `P-A8x2` (both hops), timing <= 1.03 at M1/M4,
  then an end-to-end remeasure.

## Amendment 6 (2026-09-25 ~20:40, before any D-x2-RP2 number): gates for the both-hop row-pack

**Build.** `scripts/build_dx2_b12x.py` with `DX2_DST=b12x-dx2rp2`, switched on by
`B12X_P8_DOWN_REMAINDER=rp2`. RP2 is RP plus the following, all behind
`getattr(self, "p8_input_rowpack", False)`:
- the input prologue writes `x_lo = q(x - dq(x_hi))` into token row +16;
- FC1 stages it into A row 8, with its scale word in SFA row 8;
- the MMA reads A rows q+8 from row 8, with odd-lane scales from row 8, instead of the
  broadcast duplicate;
- the FC1 epilogue folds the row q+8 accumulators into row q before the FP16 store.

M>16 paths are unchanged.

**K4 closure.** Same method as K2/K3: layers 3 and 8, four ranks summed, the Test B fit tokens.
- **Paths:** M1 (first 96 tokens) and M16 (all).
- **PASS**, per layer and path:
  - `|D_kernel(prod)/D_ref(P-A8) - 1| <= 5%`;
  - `D_kernel(RP2)/D_kernel(prod)` within +/-0.05 of the reference both-hop ratio
    `D(P-A8x2)/D(P-A8)` (Test B: layer 3 = 0.549, layer 8 = 0.869; M1 uses windows 0-1 of the
    reference per-window damage);
  - M64 and M3072 bit-identical to prod.

**A4 timing.** As A3: layers 8 and 3, rank 0, >= 60 interleaved blocks x 200 replays, M in
{1, 4, 16}. **PASS** if the median `RP2/prod` <= 1.03 at M1 and M4 on both layers.

**Next step.** If K4 and A4 pass, RP2 goes to the end-to-end measurement. Its design (window
count, arms, KV dtype) will be preregistered in a separate amendment after Window 2's FP8
results are in.

## Amendment 7 (2026-09-25 ~22:00, before any 128-window number): final 128-window run

**Gates.** K4 and A4 passed, so RP2 is the candidate.

**Windows.**
- All 128 conditional-fit windows, from `kld-cf128/verified-inputs.json`:
  - the original 32 keep their records and order;
  - 96 are scored here for the first time;
  - teacher logits come from HF revision 7c378d5f, and all 128 sha256 were verified.
- The 09-09 true-decode protocol and scorer are unchanged. The capture allow-list is widened
  to the 128 IDs.
- Fixed arm order, one fresh server each, all with FP8 MLA KV (the better cache for both
  systems):
  1. `control-fp8`: production P8 kernels.
  2. `rp2-fp8`: D-x2-RP2 (`b12x-dx2rp2`, `B12X_P8_DOWN_REMAINDER=rp2`, RP JIT volume).
     The log must show 168 `P8_DX2_ROWPACK_ACTIVE ... input_hop=1` lines.
  3. `tr3-fp8`: TR3 4bpw (`brandonmusic/GLM-5.3-Flash-tr3-4bpw` @ aba59d21), with the
     09-09 attempt-2 TR3 capture launch unchanged apart from the allow-list and capture dir.

**Primaries**, each a paired per-window difference in true-decode mean KLD with BCa95
(20,000 resamples, seed 20260902):
1. `rp2-fp8 - control-fp8`: improvement if the mean < 0 and the CI excludes 0.
2. `rp2-fp8 - tr3-fp8`, reported three ways:
   - (a) whether the CI includes 0 (not distinguishable);
   - (b) whether the whole CI lies within +/-5% of TR3's mean (equivalent at 5%);
   - (c) the same at +/-10%.

**Reported:**
- `control-fp8 - tr3-fp8`;
- per-arm means and window BCa;
- the fraction of windows lower;
- the original-32 subset compared with the earlier runs;
- per-domain means.

**Speed** (descriptive): an `rp2-fp8` production-config speed arm, same four cells and
cooling gate, compared with today's runs.

**Close:** production is restored via `systemctl --user start trellismx-klc.service` and
must finish before the 02:30 nightly job.
