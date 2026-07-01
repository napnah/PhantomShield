# AegisX Project Progress Tracking

Date: 2026-06-28

## 0. Project Direction

AegisX aims to build a real multi-party secure Transformer inference system based on the MCU protocol family. The final system should support three comparable inference paths:

- Plaintext inference: standard HuggingFace/PyTorch BERT.
- CrypTen inference: two-party MPC baseline over real Docker communication.
- MCU inference: three-role `p0 / p1 / hp` secure computation over real Docker communication.

The project should not only prove correctness of isolated protocols, but also show practical efficiency at operator level, end-to-end BERT level, and frontend demonstration level.

### Final Product Targets

The final product-level target adds service lifecycle control to the dashboard:

1. The frontend can independently control CrypTen service start/stop/status and MCU service start/stop/status.
2. When the corresponding service is running, the frontend can run BERT inference through that live service. The expected performance target is that MCU service-mode BERT inference is faster than CrypTen service-mode BERT inference under the same input and comparable Docker environment.
3. Frontend inference must be usable in practice: each inference action should either complete or return a visible partial failure/error within a bounded time, with no long-running silent delay such as an indefinite `三路比较中` state.

## 1. Current Baseline

### Implemented

- Project progress tracking rule:
  - `.cursor/rules/project-progress-tracking.mdc`
  - requires reading the relevant progress section before work and updating this document after work
- README language rule:
  - `.cursor/rules/readme-language.mdc`
  - requires the top-level `README.md` prose to be maintained in Chinese, while allowing commands, paths, identifiers, and proper nouns to remain in English
- MCU real Docker communication for `p0`, `p1`, and `hp` over TCP.
- CrypTen Docker baseline using two ranks over Gloo/TCP.
- Operator-level Docker comparison for:
  - `elemul`
  - `matmul`
  - `exp`
  - `sigmoid`
  - `gelu`
  - `softmax`
- Batch protocols for nonlinear operators.
- Communication instrumentation:
  - `send_s`
  - `recv_s`
  - `recv_wait_s`
  - `recv_read_s`
  - message counts
  - byte counts
- Tensor matmul CPU optimization:
  - fused party correction
  - fused HP matmul share
  - Rayon parallelism
  - right-matrix transpose for cache locality
- Experimental output in timestamped `experiments/` directories.
- Full BERT comparison artifacts:
  - plaintext BERT
  - CrypTen Docker 12L native baseline
  - MCU Docker 12L p0/p1/hp numerical baseline
  - earlier Python-level CrypTen/MCU-Rust paths
- Top-level `README.md`, `docker/README.md`, and `scripts/ENV.md` have been refreshed to match the current Goal 1/2/3 state, including portable Docker deployment notes, current startup modes, operator ratio table, and BERT inference efficiency table. The top-level `README.md` is now maintained in Chinese prose according to `.cursor/rules/readme-language.mdc`.

### Important Limitation

The current MCU full BERT Docker benchmark is numerically end-to-end and uses real p0/p1/hp Docker communication, but it is not yet final secure BERT because several steps still use HP-clear numerical bridges.

## 2. Main Goal 1: Operator-Level Optimization

### Goal

Optimize parallel computation and communication for every operator so that MCU is close to or faster than CrypTen whenever the MCU three-party structure does not impose an unavoidable theoretical disadvantage.

### Target Operators

- `elemul`
- `matmul`
- `exp`
- `sigmoid`
- `gelu`
- `softmax`
- protocol subroutines:
  - `sign`
  - `wrap`
  - `bicoptor`
  - `rrap`

### Current Status

| Area | Status | Notes |
|---|---|---|
| Tensor matmul protocol | Accepted on CPU Docker | BERT and large preset matmul are within the current `<2x` target; large batch 4 is near parity. |
| Nonlinear batch protocols | Implemented | Batch message paths exist for exp/sigmoid/gelu/softmax. |
| Communication timing split | Implemented | `recv_wait_s` proves much of apparent comm time is synchronization wait. |
| CPU fused tensor backend | Implemented for matmul | Party correction and HP matmul now use fused CPU kernels. |
| Tensor output timing boundary | Implemented | Protocol timing no longer counts p0/p1 verification share file writing on the critical path. |
| Bicoptor suffix generation | Optimized | Replaced per-item O(lx^2) suffix-sum construction with O(lx) construction. |
| Bicoptor batch generation | Optimized | Party-side Bicoptor message generation now uses counter-based PRG access and Rayon parallel filling. |
| CUDA backend | Prototype only, not production | Correct and callable, but forced CUDA matmul is much slower than CPU Docker in current measurements. |
| Persistent role process | Not implemented | Thread-mode batch benchmark exists as a steady-state proxy; real persistent TCP role process still open. |
| Operator-level pass/fail thresholds | Defined for current CPU Docker scope | BERT-base and large batch `1,2,4` main operators are recorded below. |

### Latest Operator Acceptance Snapshot

Date: 2026-06-28

Scope:

- Real Docker communication.
- MCU: three roles, `p0 / p1 / hp`, TCP.
- CrypTen: two ranks, Gloo/TCP.
- Preset: BERT-base-like operator shapes, batch sizes `1,2,4`.
- MCU protocol time excludes verification share file writing; write time is still reported separately.

Experiment outputs:

- Full operator matrix: `experiments/20260628_203232_docker_real_comm/summary.csv`
- Pivot ratio matrix: `experiments/20260628_203232_docker_real_comm/operator_ratio_matrix.csv`
- Exp follow-up after wrap bit-width and HP scan optimization: `experiments/20260628_205346_docker_real_comm/summary.csv`
- Exp follow-up after reducing BERT-range wrap bit-width to `33`: `experiments/20260628_210740_docker_real_comm/summary.csv`
- Large preset full operator matrix: `experiments/20260628_211821_docker_real_comm/summary.csv`
- Large preset pivot ratio matrix: `experiments/20260628_211821_docker_real_comm/operator_ratio_matrix.csv`
- Protocol subroutine benchmark and gap table: `experiments/20260628_212355_protocol_subroutines/summary.csv`
- Thread-mode batch steady-state proxy: `experiments/20260628_212639_thread_mode_batch/summary.csv`
- Forced CUDA matmul evaluation: `experiments/20260628_212819_docker_real_comm/summary.csv`
- Goal 1 technical report: `docs/20260628_goal1_operator_optimization_report.md`
- Tensor/key operators: `experiments/20260628_194149_docker_real_comm/summary.csv`
- Nonlinear operators after Bicoptor parallelization: `experiments/20260628_202325_docker_real_comm/summary.csv`
- Earlier toy full matrix: `experiments/20260628_192101_docker_real_comm/summary.csv`

| Operator | Batch 1 | Batch 2 | Batch 4 | Max | Status | Main reason |
|---|---:|---:|---:|---:|---|---|
| `elemul` | `0.86x` | `0.96x` | `0.89x` | `0.96x` | Accepted | MCU is at or faster than CrypTen for BERT-like batches. |
| `matmul` | `1.16x` | `0.95x` | `0.89x` | `1.16x` | Accepted | Meets CPU Docker target `<2x`; stretch target `<1.2x` is also met in this sweep. |
| `exp` | `1.49x` | `1.24x` | `1.73x` | `1.73x` | Accepted | BERT-range wrap bit-width `33` reduces Bicoptor payload while preserving correctness on the benchmark range. |
| `sigmoid` | `0.53x` | `0.59x` | `0.69x` | `0.69x` | Accepted | Bicoptor parallelization makes MCU faster than CrypTen in this sweep. |
| `gelu` | `0.51x` | `0.67x` | `0.78x` | `0.78x` | Accepted | MCU faster than CrypTen in this sweep. |
| `softmax` | `0.25x` | `0.28x` | `0.67x` | `0.67x` | Accepted | Batched path is faster than CrypTen for BERT-like rows. |

Interpretation:

- `elemul`, `matmul`, `exp`, `sigmoid`, `gelu`, and `softmax` are accepted for BERT-like batch sizes `1,2,4`.
- `exp` was the only non-accepted operator in the first full sweep, but follow-up optimization brought it below `2x` CrypTen for batch sizes `1,2,4`.
- The previous `sigmoid/gelu` bottleneck was resolved by counter-based parallel Bicoptor generation; `exp` is accepted but remains the clearest communication-heavy nonlinear operator.
- The `large` preset sweep is complete for main operators; all six are accepted under the CPU Docker `<2x` target.

### Subgoals

#### 1.1 Establish Operator Benchmark Matrix

Status: Complete for current CPU Docker main-operator scope; protocol subroutine naming clarified

Tasks:

- Define standard shapes:
  - toy: small correctness smoke test
  - BERT-base: `seq=128`, `hidden=768`, `heads=12`, `ffn=3072`
  - large: `seq=128`, `hidden=1024`, `heads=16`, `ffn=4096`
- Define batch sizes:
  - `1`
  - `2`
  - `4`
  - optional stress tests: `8`, `16`
- For each operator, record:
  - MCU median time
  - CrypTen median time
  - plaintext Torch median time
  - `MCU / CrypTen`
  - `MCU / Plain`
  - `CrypTen / Plain`
  - communication breakdown
  - local compute breakdown
  - correctness error

Acceptance:

- Every operator has at least one toy, one BERT-base, and one larger-shape benchmark.
- Every benchmark row includes correctness verification.
- CSV output is generated under `experiments/<timestamp>_.../`.

Latest completion note:

- Main operators now have BERT-base and large preset Docker results for batch sizes `1,2,4`.
- Protocol subroutines `sign`, `wrap`, `sign-bicoptor`, and `wrap-bicoptor` have MCU real-Docker throughput and correctness rows.
- No independent `rrap` protocol/API was found in the current project or extracted paper text. The relevant repeated truncation, random shuffle, masking, and reshare steps are implemented inside Bicoptor sign and measured through `sign-bicoptor` / `wrap-bicoptor`.

#### 1.2 Matmul Optimization

Status: Accepted for BERT-base and large CPU Docker; CUDA production path blocked

Completed:

- Tensor-level matmul protocol.
- Fused CPU correction kernel for party side.
- Fused CPU HP share kernel.
- Parallel row-level execution via Rayon.
- Right-matrix transpose for better memory locality.
- Communication optimization:
  - HP parallel receive.
  - HP parallel send.
  - large payload segmented writes.
- Protocol timing excludes verification share file writing.

Latest result:

- BERT-base batch=1 shape `[128,768]x[768,768]`: accepted under `<2x`.
- Large preset max across batch sizes `1,2,4`: `1.27x`.
- Forced CUDA BERT matmul used CUDA kernels with no fallback, but was `7.39x` CrypTen and much slower than CPU MCU.

Remaining:

- Add blocked/tiled matmul kernel instead of only transposed row dot products.
- Tune `MCU_CPU_PAR_MIN_OPS` by shape.
- CUDA production path needs redesigned kernels and persistent device tensors before it is worth pursuing further.
- Explore persistent GPU workspace only after end-to-end tensor lifetime can stay on device.

Acceptance:

- BERT-base matmul should be at most `2x` CrypTen on CPU Docker.
- Stretch target: BERT-base matmul should be at most `1.2x` CrypTen.
- Any unavoidable gap must be explained by protocol traffic or three-party structure, not implementation overhead.

#### 1.3 Nonlinear Operator Optimization

Status: Accepted for BERT-base and large batch sizes `1,2,4`

Tasks:

- Re-run full nonlinear matrix after communication optimization.
- Identify whether bottleneck is:
  - message count
  - `recv_wait_s`
  - local nonlinear computation
  - HP fan-in/fan-out
- Ensure `exp`, `sigmoid`, `gelu`, `softmax` use fully batched paths in real Docker mode.
- Add fused batch execution for common BERT patterns:
  - attention softmax over all heads/rows
  - FFN GeLU over full hidden tensor

Completed:

- Real Docker mode uses batched paths for `exp`, `sigmoid`, `gelu`, and `softmax`.
- Bicoptor suffix-sum generation was reduced from O(lx^2) to O(lx).
- Nonlinear protocol timing no longer includes final output string formatting and verification-file writing.
- Party-side Bicoptor batch generation now uses counter-based PRG random access and Rayon parallel filling.
- `wrap_bicoptor_batch` combines hi/lo sign checks into one larger Bicoptor batch.
- Wrap fixed-point comparison bit-width defaults to `33` for the current BERT-like benchmark range, with `MCU_WRAP_FIXED_LX` override for wider numeric ranges.
- HP-side Bicoptor zero-detection scan is parallelized for large batches.
- Party-side Bicoptor value generation no longer allocates a temporary truncation vector for every item.
- `exp` batch PRG/share generation has an optional counter-based parallel path controlled by `MCU_EXP_PAR_MIN`, but it is not enabled for BERT batch sizes `1,2,4` by default because Docker CPU thread scheduling and memory pressure made it slower in measurement.

Latest result:

- `softmax`: accepted with max `0.67x` CrypTen across batch sizes `1,2,4`.
- `gelu`: accepted with max `0.78x` CrypTen across batch sizes `1,2,4`.
- `sigmoid`: accepted with max `0.69x` CrypTen across batch sizes `1,2,4`.
- `exp`: accepted with max `1.73x` CrypTen across batch sizes `1,2,4`.
- Large preset latest max ratios:
  - `exp`: `1.51x`
  - `sigmoid`: `0.44x`
  - `gelu`: `0.50x`
  - `softmax`: `0.23x`

Current bottleneck:

- The former party-side Bicoptor/wrap generation bottleneck is largely resolved for `sigmoid` and `gelu`.
- Standalone `exp` still has a structural wrap-detection dependency and sends large Bicoptor payloads. In the latest batch=4 run, HP received about `447 MB`, down from about `535 MB` before the `lx=33` tuning; the remaining gap is mainly payload transfer plus synchronization, not scalar `exp()` compute.
- Remaining risk is numeric-range generality for `lx=33`; wider model activation ranges must use `MCU_WRAP_FIXED_LX` or an explicit range check.

Acceptance:

- `softmax` should remain faster than CrypTen for BERT-like rows if current trend holds.
- `sigmoid` and `gelu` should be close to or faster than CrypTen on BERT-like tensors.
- `exp` should be investigated separately because it may carry more protocol-specific communication.

#### 1.4 Elemul and Low-Level Multiplication

Status: Accepted for BERT-base batch=1; small-shape overhead still high

Tasks:

- Confirm `elemul` uses vector protocol everywhere.
- Benchmark larger tensor lengths where Docker startup and fixed overhead are amortized.
- Add persistent role benchmark to measure steady-state elementwise throughput.

Latest result:

- BERT-base batch=1 length `[98304]`: MCU/CrypTen = `1.13x`.
- Toy shapes remain misleading because fixed Docker/process synchronization dominates.

Acceptance:

- For large tensors, `elemul` should approach CrypTen throughput unless three-party HP fan-in/fan-out dominates.
- If slower, report whether gap comes from bytes, sync wait, or local arithmetic.

#### 1.5 Theoretical Gap Analysis

Status: Implemented for measured operators; `rrap` treated as Bicoptor-internal terminology unless a separate paper definition is supplied

Tasks:

- For each operator, write a short analysis of expected communication:
  - number of parties
  - rounds
  - bytes sent by p0/p1/hp
  - critical path
- Mark each gap as:
  - implementation gap
  - protocol/architecture gap
  - measurement artifact

Acceptance:

- Every operator benchmark has a matching explanation for any `MCU / CrypTen > 1.5x`.

Latest output:

- Gap table: `experiments/20260628_212355_protocol_subroutines/operator_theoretical_gap.csv`
- Current `>1.5x` case is mainly `exp` at large batch 4 (`1.51x`), explained by Bicoptor payload transfer and HP receive/read time.
- The earlier `rrap` checklist item has been clarified: no standalone `rrap` definition was found locally, and the implemented Bicoptor sign already covers the repeated truncation/randomization/reshare path that was being referenced.

## 3. Main Goal 2: Full BERT Docker Inference

Decision status: Closed as Goal2 v1 on 2026-06-29. The completed scope is numerical end-to-end Docker BERT inference and CrypTen comparison. Final secure replacement of HP-clear bridges is intentionally deferred to a later security-focused goal.

### Goal

Implement full BERT inference launched through Docker and compare end-to-end inference efficiency against CrypTen two-party Docker inference. The target is for MCU Docker inference to reach or exceed CrypTen two-party inference efficiency.

### Current Status

| Area | Status | Notes |
|---|---|---|
| Python full BERT benchmark | Implemented | Compares plaintext, CrypTen, MCU-Rust extension. Not real Docker communication. |
| MCU Docker full BERT | Goal2 v1 complete | `bert_session` runs persistent p0/p1/hp Docker TCP 12-layer SST-2 BERT with exported embedding, attention mask, Q/K/V/O/FFN weights, all encoder biases, LayerNorm params, pooler, classifier, logits, and probabilities. HP-clear bridges make this a numerical baseline, not a final secure implementation. |
| CrypTen Docker full BERT | Baseline implemented | `CRYPTEN_OP=bert_full` runs two Docker ranks over Gloo/TCP with embedded SST-2 checkpoint. Native nonlinear path is the correctness baseline; legacy 2Quad remains optional. |
| BERT real weights | Available | `bert-base-uncased` and SST-2 checkpoint exist locally. |
| Frontend BERT comparison | Implemented for current paths | Dashboard can call process, cold Docker, warm Docker, and service-mode Docker paths. CrypTen service is fully persistent; MCU service is wrapper-persistent and still exports shares/reconnects internally per request. |

Latest output:

- Technical report: `docs/20260628_goal2_bert_docker_testing_report.md`
- CrypTen native full-path baseline: `experiments/20260628_223612_docker_bert_full/summary.csv`
- CrypTen native result: 10 samples, 12 layers, max sequence length 16. Plaintext host avg `0.0486s/sample`, CrypTen Docker native avg `11.6562s/sample`, rank-level full run about `117.8s`, accuracy/top-1 agreement `0.90`, mean JS `4.16e-3`.
- CrypTen legacy 2Quad result: `experiments/20260628_223028_docker_bert_full/summary.csv`, 10 samples, avg `2.0374s/sample`, accuracy/top-1 agreement `0.60`, mean JS `1.77e-2`.
- MCU numerical fix and final Goal2 run:
  - fixed missing Q/K/V bias export and consumption;
  - changed MCU attention mask penalty from HF-style `-10000` to protocol-domain-safe `-80`, because the real exponential protocol works modulo `MOD=256` and `-10000` wrapped instead of masking padding;
  - local validation: `experiments/20260629_175452_mcu_bert_session_smoke/summary.csv`;
  - Docker validation: `experiments/20260629_180301_mcu_bert_session_docker/summary.csv`;
  - accuracy report: `experiments/20260629_180736_mcu_bert_accuracy/summary.csv`;
  - combined Goal2 comparison: `experiments/20260629_180900_goal2_docker_bert_comparison/summary.csv`.
- MCU Docker result: 10 samples, 12 layers, max sequence length 16, real p0/p1/hp Docker TCP, critical role `66.80s`, avg `6.68s/sample`, accuracy/top-1 agreement with plaintext `1.00`, mean JS `2.49e-4`.
- Current MCU/CrypTen native latency ratio is about `0.57x` by average per-sample time (`6.68s / 11.66s`), while MCU has better top-1 agreement on this 10-sample check.
- Storage update: CrypTen image now embeds only `checkpoints/bert-sst2`, avoiding F: drive model bind-mount failures and repeated C: temp model staging.
- Storage migration update: Docker Desktop WSL data was moved from `C:\Users\31248\AppData\Local\Docker\wsl` to project-local ignored storage `F:\AI_Agent\MCU-transformer\.local\docker-desktop-data\wsl`, with a junction left at the original C: path. C: free space increased from about `8 GB` to about `76 GB`, Docker images remained visible, and `.local/` is ignored by git. Project-local cache directories were also created for pip/HuggingFace/Torch/temp.

### Required Architecture

```mermaid
flowchart LR
    UserInput["Input Text"]
    Tokenizer["Tokenizer / Input Builder"]

    subgraph MCU["MCU Docker BERT"]
        P0["p0: data/input share"]
        P1["p1: model/share side"]
        HP["hp: helper"]
        P0 <--> HP
        P1 <--> HP
    end

    subgraph CrypTen["CrypTen Docker BERT"]
        R0["rank0"]
        R1["rank1"]
        R0 <--> R1
    end

    Tokenizer --> P0
    Tokenizer --> R0
    P0 --> MCUOut["MCU logits/probs"]
    P1 --> MCUOut
    R0 --> CrypTenOut["CrypTen logits/probs"]
    R1 --> CrypTenOut
```

### Subgoals

#### 2.1 CrypTen Docker `bert_full`

Status: Baseline implemented, native path accepted as current correctness baseline

Tasks:

- Completed: `docker/crypten_rank_bench.py` supports `CRYPTEN_OP=bert_full`.
- Completed: CrypTen image embeds the SST-2 checkpoint and no longer needs model bind mounts for the default run.
- Completed: fixed input JSON is supported.
- Completed: output includes latency, prediction, probability vector, accuracy inputs, and security profile.
- Completed: added `CRYPTEN_BERT_NONLINEAR=native` to improve agreement with plaintext.
- Remaining: remove plaintext/reconstruction points and reduce native-mode latency.

Acceptance:

- Met: `docker compose` can run `crypten-r0` and `crypten-r1` for `bert_full` with embedded checkpoint.
- Met: results are written to shared output as JSON and CSV by `experiments/docker_bert_full/run_docker_bert_comparison.py`.
- Partially met: native mode reaches `0.90` top-1 agreement on the current 10-sample check, but still has divergence from plaintext and is much slower than legacy 2Quad.

#### 2.2 MCU Docker Full BERT Session

Status: Numerical end-to-end Docker benchmark implemented; final secure protocols still open

Tasks:

- Completed: persistent local and Docker p0/p1/hp BERT sessions.
- Completed: real_io share export for input embedding, attention mask, Q/K/V/O/FFN weights, Q/K/V/O/FFN biases, LayerNorm params, pooler, classifier, and labels.
- Completed: full numerical state flow:
  - embedding/input preparation
  - Q/K/V/O linear projections with Q/K/V/O biases
  - attention score matmul
  - softmax feedback
  - value matmul
  - residuals and LayerNorm
  - FFN in/out and GeLU feedback
  - pooler tanh
  - classifier logits/probabilities/predictions
- Completed: `experiments/docker_bert_full/compare_mcu_docker_accuracy.py` writes per-sample and summary accuracy/divergence CSV against plaintext.
- Completed: fixed two numerical mismatches:
  - Q/K/V biases were missing from MCU export/session;
  - HF `-10000` attention mask was invalid for MCU real exponential modulo `MOD=256`; MCU now uses `-80`, which masks padding without wraparound.
- Not completed yet:
  - wrap-correct secure rescale/truncation after matmul; `local` can be wrong around ring wrap boundaries, while `hp_clear` is numerically useful but insecure because HP reconstructs values;
  - secure fixed/real conversion for softmax and GeLU feedback; current HP-clear bridges are numerical baselines only;
  - secure attention-mask handling policy in the final threat model; current mask is exported as public metadata;
  - secure LayerNorm strategy; current LayerNorm is HP-clear numerical baseline only;
  - secure pooler tanh and final probability reveal.

Acceptance:

- Met for numerical Goal2 benchmark: Docker three-role TCP session completes 12-layer real_io BERT and writes prediction/probability outputs.
- Met for current performance target: MCU Docker avg `6.68s/sample` vs CrypTen Docker native `11.66s/sample` on the same 10-sample SST-2 check.
- Met for current accuracy check: MCU Docker top-1 agreement with plaintext is `1.00`, mean JS `2.49e-4`.
- Not met for final security: HP-clear rescale/conversion/LayerNorm/tanh/reveal remain security blockers.

#### 2.3 Security Boundary Audit

Status: Updated audit recorded; numerical Goal2 path is not final secure BERT

Tasks:

- Completed initial record in `docs/20260628_goal2_bert_docker_testing_report.md`.
- Current plaintext/reconstruction points:
  - tokenization and embedding lookup;
  - both CrypTen ranks load the model checkpoint;
  - legacy 2Quad mode reconstructs scores before `two_quad`;
  - CrypTen native LayerNorm variance inverse reconstructs variance and attention/pooler points are partly reconstructed;
  - MCU HP-clear rescale reconstructs fixed-point matmul outputs at HP;
  - MCU HP-clear fixed/real bridges reconstruct attention scores, GeLU inputs/outputs, pooler tanh input/output, and classifier logits at HP;
  - MCU HP-clear LayerNorm reconstructs hidden rows and LayerNorm parameters at HP.
- Remaining: decide which points are acceptable for the target threat model and replace the rest with secure protocols.

Acceptance:

- A `docs/security_boundary.md` or equivalent section exists.
- Every plaintext operation in full BERT inference is explicitly classified:
  - public input
  - allowed leakage
  - temporary prototype
  - security blocker

#### 2.4 Persistent Process and Request Protocol

Status: Warm Docker service v1 implemented; CrypTen persistent rank service v2 implemented; MCU persistent role-wrapper service v2 implemented; true MCU in-process persistent model/session state still needed

Tasks:

- Completed for one inference: p0/p1/hp connections stay open during full BERT inference.
- Completed v1: avoid starting containers for every dashboard request by using `docker/docker-compose.warm.yml` plus `experiments/docker_bert_full/warm_docker_service.py`.
- Completed v1: backend can manage warm containers via `POST /api/bert/docker-service` and can call `launch="docker_warm"` for CrypTen and MCU BERT comparison.
- Completed v1: timing now reports `container_start_s=0.0` for warm requests and separates total host wall time from protocol/critical-role time.
- Completed v2 for CrypTen: `docker/crypten_persistent_service.py` keeps both CrypTen rank processes alive, initializes CrypTen once, loads the BERT model/tokenizer once, and processes repeated request JSON files from the warm shared volume.
- Completed v2 API for CrypTen: backend supports `launch="docker_service"` for CrypTen; `POST /api/bert/docker-service` supports `start_crypten`, `stop_crypten`, and `crypten_status`.
- Completed v2 wrapper for MCU: Rust `bert_session` now has `service-hp`, `service-p0`, and `service-p1` role loops that stay alive, watch request files, and write per-role completion files.
- Completed v2 API for MCU: backend supports `launch="docker_service"` for MCU; `POST /api/bert/docker-service` supports `start_mcu`, `stop_mcu`, and `mcu_status`.
- Completed v2.1 data-preparation cache for MCU:
  - static model/checkpoint shares are cached under `.local/docker_warm/bert_shares/_model_cache/`;
  - per-request input share export is handled by a host-side persistent input exporter that keeps tokenizer/model loaded;
  - Rust `bert_session` accepts `--model-share-dir` and falls back from request-local shares to cached model shares.
- Remaining for true MCU persistent session: current service wrapper still calls the existing per-request session internally, so protocol TCP connection setup and request-local share file reads still happen per request.
- Add a true in-process request protocol:
  - initialize model/session
  - accept repeated inference input without restarting rank/role binaries
  - run operator/layer requests over the same rank/role process lifetime
  - finalize output
- Separate:
  - startup time
  - model loading time
  - protocol execution time
  - output reconstruction time

Acceptance:

- Met for v1: end-to-end warm service timing excludes one-time Docker container startup after `warm_docker_service.py start`.
- Met for v1: warm-run inference can be repeated multiple times in the same containers.
- Met for CrypTen v2: warm-run inference can be repeated in the same rank processes with model/tokenizer already loaded.
- Met for MCU v2 wrapper: repeated requests can be handled by the same `hp/p0/p1` wrapper processes.
- Met for MCU v2.1 data preparation: model share export and input embedding/share export are no longer repeated as heavy subprocess work per request.
- Not met for final MCU service: inner TCP protocol state and Rust-side loaded tensor/model state are not yet persistent across requests.

Latest validation:

- Warm containers started successfully on 2026-06-30 with project `aegisxwarm`: `crypten-warm-r0`, `crypten-warm-r1`, `mcu-warm-hp`, `mcu-warm-p0`, and `mcu-warm-p1`.
- CrypTen warm direct run: `.local/docker_warm/out/crypten_20260630_125348_140184/result.json`, protocol latency about `15.34s`, host wall time about `44.58s`.
- MCU warm direct run: `.local/docker_warm/out/mcu_20260630_125445_108903/result.json`, critical role about `22.26s`, host wall time about `23.87s`.
- Backend API warm CrypTen smoke: `launch=docker_warm`, success `1/1`, artifact `.local/docker_warm/out/crypten_20260630_130004_138069`.
- Backend API warm MCU smoke: `launch=docker_warm`, success `1/1`, artifact `.local/docker_warm/out/mcu_20260630_130212_259550`; host wall time stayed high because share export/preparation remains per request.
- CrypTen persistent v2 startup: `warm_docker_service.py start-crypten-service`, ready in about `8.10s`.
- CrypTen persistent v2 repeated requests:
  - `.local/docker_warm/out/crypten_service_20260630_132412_826514/result.json`, request_count `1`, latency about `18.86s`.
  - `.local/docker_warm/out/crypten_service_20260630_132456_501246/result.json`, request_count `2`, latency about `18.21s`.
  - Backend API `launch=docker_service`, artifact `.local/docker_warm/out/crypten_service_20260630_132505_404281`, success `1/1`, model/load/container startup excluded from per-request timing.
- MCU persistent wrapper v2 validation:
  - Rust `cargo build --release --bin bert_session` passed on 2026-06-30.
  - `warm_docker_service.py start-mcu-service` started ready `hp/p0/p1` wrapper roles in about `2.08s`.
  - Direct repeated requests succeeded: `.local/docker_warm/out/mcu_service_20260630_141154_429051/result.json` and `.local/docker_warm/out/mcu_service_20260630_141916_763716/result.json`, each about `25.5s` service elapsed at max sequence length `4`.
  - Backend API `launch=docker_service`, mode `mcu_rust`, succeeded with artifact `.local/docker_warm/out/mcu_service_20260630_142742_930751`.
  - Follow-up backend service smoke at max sequence length `4`: CrypTen `.local/docker_warm/out/crypten_service_20260630_144627_526626`, protocol about `8.70s`; MCU `.local/docker_warm/out/mcu_service_20260630_144637_388932`, service elapsed about `22.40s` but host wall-clock about `397s`.
  - v2.1 cache/exporter optimization:
    - static model share cache added via `--static-only` export and Rust `--model-share-dir` fallback;
    - persistent host input-share exporter added in `experiments/docker_bert_full/mcu_input_share_service.py`;
    - direct MCU service requests at max sequence length `4` now show `input_share_export_s` about `0.12s` instead of about `13s`;
    - validated artifacts: `.local/docker_warm/out/mcu_service_20260630_161406_070615/result.json` and `.local/docker_warm/out/mcu_service_20260630_161427_537642/result.json`;
    - end-to-end command wall-clock is now close to the 20-21s service execution time instead of hundreds of seconds.
  - Important limitation: this direction remains useful, but the next meaningful step is moving the inner TCP/session state and loaded model tensors into a real persistent Rust request loop.

#### 2.5 End-to-End Benchmark

Status: Completed for numerical Docker benchmark; secure benchmark remains future work

Tasks:

- Completed: created `experiments/docker_bert_full/run_docker_bert_comparison.py`.
- Completed: compares plaintext host baseline and CrypTen Docker `bert_full` in native and legacy modes.
- Completed: outputs CSV summary, JSON per sample, rank logs, accuracy, latency, top-1 agreement, KL/JS/L1/L2/max-abs divergence.
- Completed: MCU local and Docker BERT-session benchmarks write module timing, role timing, summary, and result JSON.
- Completed: MCU-vs-plaintext accuracy comparison writes per-sample and summary CSV.
- Completed: combined Goal2 Docker comparison in `experiments/20260629_180900_goal2_docker_bert_comparison/summary.csv`.

Acceptance:

- Met for CrypTen baseline: 10-sample 12-layer native benchmark completed.
- Met for MCU numerical benchmark: 10-sample 12-layer Docker run completed with real p0/p1/hp TCP.
- Met for current efficiency target: MCU is about `0.57x` CrypTen native latency on this benchmark.
- Not met for final security: the benchmark still relies on HP-clear numerical bridges.

## 4. Main Goal 3: Frontend Integration

Decision status: Goal3 complete on 2026-06-30. The dashboard uses the original `index.html` as the default frontend entry. `index_new.html` remains available as an alternate/newer page with the medical, finance, and sentiment three-way comparison flow. Sentiment delegates to the BERT comparison path; medical and finance use the shared scenario comparison API. Docker launch calls the existing real CrypTen two-rank and MCU p0/p1/hp Docker runners where available; MCU remains clearly labeled as an HP-clear numerical prototype until a later security goal replaces those bridges.

### Goal

Connect Docker inference paths to the frontend so that users can compare plaintext, CrypTen, and MCU inference in both efficiency and execution form.

### Current Status

| Area | Status | Notes |
|---|---|---|
| Frontend UI | Complete | `dashboard/frontend/index.html` is the original default dashboard; `index_new.html` is retained as an alternate page where medical, finance, and sentiment share the launch-mode selector and compact three-path comparison result grid. |
| Backend semantic inference | Complete | Existing Python/process path remains available for `plaintext`, `crypten`, and `mcu_rust`. |
| Docker inference backend | Complete | `/api/bert/infer`, `/api/bert/docker-infer`, `/api/bert/compare`, `/api/bert/docker-service`, and `/api/scenario/compare` expose cold Docker and warm Docker v1 comparison paths; sentiment uses BERT orchestration and medical/finance use scenario-level comparison. |
| Frontend service lifecycle control | Basic implemented | Default dashboard and `index_new.html` expose CrypTen/MCU service start/stop/status controls through `/api/bert/docker-service`. |
| Docker logs/status | Basic complete | Backend returns recent structured role/rank logs and status after each single or comparison run; live streaming is not implemented. |
| Benchmark history | Basic complete | Backend records recent dashboard BERT runs in `.local/dashboard/bert_history.jsonl`; frontend history browser is not implemented. |

### Required User Experience

The frontend should clearly distinguish the three inference forms:

- Plaintext:
  - single process
  - full data/model visible
  - fastest baseline
- CrypTen:
  - two Docker ranks
  - Gloo/TCP communication
  - MPC baseline
- MCU:
  - three Docker roles
  - p0/p1/hp TCP communication
  - protocol logs and communication breakdown

### Subgoals

#### 3.1 Backend Docker Orchestrator API

Status: Complete

Tasks:

- Add backend endpoint:
  - `POST /api/bert/infer`
  - `POST /api/bert/compare`
  - `POST /api/bert/docker-infer`
  - `POST /api/bert/docker-service`
  - `GET /api/bert/docker-status`
  - `GET /api/bert/docker-logs`
  - `GET /api/bert/docker-benchmark`
- The backend should be able to:
  - completed: start the existing Docker runners through subprocess;
  - completed: run one BERT inference request for CrypTen Docker or MCU Docker;
  - completed: start/stop/status warm Docker service containers and run `launch=docker_warm`;
  - completed: collect output JSON and normalize prediction/probabilities/latency/logs;
  - completed: return recent structured logs and status;
  - optional follow-up: true live log streaming and persistent warm Docker services for lower cold-start latency.

Acceptance:

- Frontend can trigger Docker MCU and Docker CrypTen inference without manual command-line execution when Docker Desktop/daemon and required images are available.
- Validation note: FastAPI process-mode three-path smoke passed on 2026-06-30. Full Docker three-path compare also passed after starting Docker Desktop: plaintext positive, CrypTen Docker positive, MCU Docker positive. Output artifacts:
  - CrypTen: `experiments/20260630_101258_docker_bert_full/summary.csv`
  - MCU: `experiments/20260630_101402_mcu_bert_session_docker/summary.csv`
  - Note: the CrypTen runner's standalone summary still contains its historical `mcu_docker=blocked` placeholder row; the dashboard comparison result uses the separate MCU session artifact above for the actual MCU Docker result.
- Warm service validation note: `launch=docker_warm` passed backend smoke for CrypTen and MCU on 2026-06-30. Warm v1 reuses containers but still starts role/rank binaries per request.
- Persistent service validation note: `launch=docker_service` passed backend smoke for CrypTen and MCU on 2026-06-30. CrypTen v2 reuses containers, rank processes, and loaded BERT model/tokenizer. MCU v2 reuses containers and `hp/p0/p1` wrapper processes, but still reconnects and reloads shares inside each request. The default dashboard now exposes `Docker 服务` for sentiment BERT single-path runs, and `index_new.html` exposes it for three-path comparison.

#### 3.2 Result Model

Status: Complete

Tasks:

- Common result schema implemented by `dashboard/backend/bert_orchestrator.py`:

```json
{
  "mode": "plaintext | crypten | mcu_rust",
  "launch": "process | docker",
  "label": "positive",
  "prediction": 1,
  "probabilities": [0.1, 0.9],
  "distribution": [{"label": "positive", "prob": 90.0}],
  "latency": {
    "protocol_s": 0.0,
    "total_s": 0.0
  },
  "logs": [],
  "artifact_dir": "experiments/...",
  "security_note": ""
}
```

Acceptance:

- Plaintext, CrypTen, and MCU outputs all conform to the same schema.
- Frontend does not need custom parsing per backend mode.

#### 3.3 Frontend Comparison View

Status: Complete

Tasks:

- Completed:
  - preserved the existing dashboard layout;
  - added a BERT launch selector with `process` and `Docker`;
  - routed sentiment BERT requests to `/api/bert/infer`;
  - display method, prediction, probability distribution, latency, attention tokens, and structured logs from the unified result.
  - added `POST /api/bert/compare` for sequential three-path comparison over the same text;
  - added a compact frontend `比较三路` button/result grid for plaintext, CrypTen, and MCU;
  - kept `index_new.html` as an alternate three-way comparison page, while restoring the original `index.html` as the default frontend entry;
  - added `POST /api/scenario/compare` so medical, finance, and sentiment all use the same three-path frontend flow;
  - replaced stale comparison cards with an explicit in-progress state while a new comparison is running;
  - comparison mode supports partial success, so Docker environment failures are shown per path instead of hiding successful plaintext/process results.
- Optional follow-up:
  - richer side-by-side charts and persisted benchmark selection;
  - detailed communication byte/message visualization beyond the existing log panel.
- Show:
  - prediction
  - probability distribution
  - latency
  - communication bytes/messages
  - role/rank topology
  - security boundary notes
- Add visual distinction:
  - single-process plaintext
  - two-party CrypTen
  - three-party MCU

Acceptance:

- A user can run the same text through all three modes and compare results on one screen in both process mode and Docker mode.
- The UI does not imply that Python extension MCU and Docker MCU are the same execution form.
- Validation note: process-mode smoke passed on 2026-06-30 for `medical`, `finance`, and `sentiment`, each returning `plaintext`, `crypten`, and `mcu_rust` with `3/3` success. Browser check confirmed `index_new.html` renders three comparison cards; default `/` has since been restored to the original `index.html` by user preference.

#### 3.4 Benchmark History

Status: Backend complete; frontend viewer optional follow-up

Tasks:

- Completed: Docker runners still write full artifacts under `experiments/`.
- Completed: dashboard backend records recent UI runs in `.local/dashboard/bert_history.jsonl`.
- Completed: `/api/bert/docker-benchmark` returns recent dashboard run summaries.
- Optional follow-up: frontend history picker and summary chart.

Acceptance:

- Backend can return recent run summaries. A richer frontend history picker remains optional follow-up work, not required for Goal3 completion.

#### 3.5 Frontend Service Lifecycle Control

Status: Basic implemented; performance target still active

Final target:

- The frontend exposes independent controls for CrypTen service:
  - start
  - stop
  - status
- The frontend exposes independent controls for MCU service:
  - start
  - stop
  - status
- When CrypTen service is running, the frontend can issue BERT inference with `launch="docker_service"` and `mode="crypten"`.
- When MCU service is running, the frontend can issue BERT inference with `launch="docker_service"` and `mode="mcu_rust"`.
- The frontend comparison view should report service-mode latency for both paths and make the target explicit: MCU service-mode BERT should be faster than CrypTen service-mode BERT for comparable inputs and Docker conditions.

Current backend support:

- Implemented: `POST /api/bert/docker-service` already supports service actions through `bert_orchestrator.docker_service`.
- Implemented: CrypTen service v2 can keep rank processes and model/tokenizer loaded.
- Implemented: MCU service v2.3 can keep role wrapper processes, static model shares, host input-share exporter, Rust-side model-share cache, and fused Q/K/V projection warm.

Completed frontend work:

- Added CrypTen and MCU service start/stop/status controls to `dashboard/frontend/index.html`.
- Added the same controls to `dashboard/frontend/index_new.html`.
- `index_new.html` computes and displays `MCU latency / CrypTen latency` when both service-mode results are available.
- Added automatic service status polling and service-offline warnings/blocking for `launch="docker_service"`.
- Fixed MCU service health checks so stale `.ready` files no longer hide a crashed input exporter.
- Added live communication-event prompts in the frontend log panels:
  - default `index.html` single-path inference now streams staged UI/API/CrypTen/MCU events while a request is running, then keeps backend-returned logs after completion;
  - `index_new.html` three-path comparison now streams staged Plaintext, CrypTen, and MCU communication events while waiting, then replaces them with real backend logs after completion.
- Edge validation on 2026-06-30:
  - tested with Edge profile `rbmm8dar` through project-local user data under `.local/edge_user_data/`;
  - restarted the stale dashboard backend so the current `/api/bert/docker-service` route was loaded;
  - confirmed both default `index.html` and alternate `index_new.html` render service controls and show CrypTen/MCU status as `运行中`;
  - confirmed `index_new.html` medical and finance three-path comparisons complete in service mode with no browser network errors;
  - Follow-up root cause: the slow-looking sentiment run was a failed MCU service request, not a normal multi-minute inference. `/api/scenario/compare` used `max_seq_len=16`, while the already-running persistent CrypTen/MCU services had been started with `max_seq_len=4`. CrypTen ignored the per-request length and completed with service `seq=4`; MCU exported 16-token input shares but the Rust service expected 4-token shares, causing `hidden.bin` length mismatch and role panic. Because the wrapper panic did not write role `.error` files, the backend waited for its long service timeout and the frontend stayed on `三路比较中`.
  - Fixed: persistent service config is now checked against requested `max_seq_len/layers`; CrypTen and MCU service queues are cleaned on service start; MCU share length mismatch returns an error instead of panicking; service waits are bounded by `AEGISX_SERVICE_REQUEST_TIMEOUT_S` default `45s`.
  - Retest passed after rebuilding `phantomshield-mcu:bert-session-fixed`: API sentiment service compare returned `3/3` in `45.0s`; Edge `index_new.html` sentiment service compare returned `3/3` in `38.6s`; default `index.html` MCU service inference returned in `15.7s`. No browser network errors and no stuck `三路比较中` state.
  - Live-log retest passed: Edge screenshots confirm `index_new.html` shows staged communication events during service comparison and real logs after completion; default `index.html` shows staged MCU service events during single-path inference and backend-returned logs after completion.
  - Output screenshots/results: `.local/frontend_checks/edge_index_after_restart_sentiment_service.png`, `.local/frontend_checks/edge_index_new_medical_compare.png`, `.local/frontend_checks/edge_index_new_finance_compare.png`, `.local/frontend_checks/edge_retest_sentiment_after_compare.png`, `.local/frontend_checks/edge_retest_index_after_mcu_single.png`, `.local/frontend_checks/edge_retest_sentiment_after_latency_fix.json`, `.local/frontend_checks/api_sentiment_compare_after_latency_fix.json`.

Latest service-mode benchmark:

- Date: 2026-06-30.
- Scope: Docker warm services, `max_seq_len=4`, 12 BERT layers, one SST-like text.
- CrypTen persistent service: `4.79s` protocol elapsed in the latest stable run.
- MCU persistent service after model-share cache, fixed-sleep removal, and fused Q/K/V projection: best stable observed `6.50s` service elapsed; later Docker/CPU contention runs varied around `12.46s`.
- Ratio: best observed `MCU / CrypTen = 1.31x`; later contention run around `1.48x`. This improved from about `4.07x` before the service fixes, but is still not below `1.0x`.
- Bottleneck: no longer container startup or model-share export. The current run still performs request-local inner TCP sessions and 338 MCU matmul protocol calls; HP receives about `1.56GB` per request, and HP communication/receive time dominates under Docker contention.
- Paused direction: HP-side static right-matrix cache was prototyped and reverted because reusing masked model weights requires mask/correction lifecycle redesign; using a naive cache is not a clean security-aligned optimization.

Remaining work:

- Continue optimizing MCU service-mode BERT until `MCU latency / CrypTen latency < 1.0` is consistently achieved under comparable Docker conditions.
- Highest-promise next step: fuse BERT-level protocol calls, especially Q/K/V projection and per-head attention score/value paths, to reduce request round trips and HP payloads.
- Medium-promise next step: replace the wrapper-style service with a true persistent inner TCP/session loop so HP/party sockets and request state do not reconnect per request.
- Do not reintroduce static masked-weight reuse unless the PRG mask/correction lifecycle is redesigned and documented.

Acceptance:

- Basic met: a user can start/stop/query CrypTen service from the frontend.
- Basic met: a user can start/stop/query MCU service from the frontend.
- Basic met: a user can run side-by-side service-mode BERT comparison and see whether `MCU latency / CrypTen latency < 1.0`.
- Basic met: frontend inference actions now complete or return a visible partial failure within a bounded time; the previously observed long `三路比较中` delay was fixed and retested in Edge.
- Not fully met: the performance target is not reached yet; MCU still needs BERT-level fused tensor protocols and true persistent inner TCP/session state.

## 5. Roadmap

### Phase 1: Operator Closure

Priority: Highest

Deliverables:

- Re-run operator matrix after latest matmul optimization.
- Add operator status table with accepted ratios.
- Identify remaining slow operators.
- Optimize `exp`, `sigmoid`, `gelu`, and `softmax` where needed.

Exit criteria:

- Every operator has a documented performance status.
- Any operator slower than CrypTen has a measured bottleneck and next action.

### Phase 2: Docker Full BERT Prototype

Priority: Highest

Deliverables:

- CrypTen Docker `bert_full`.
- MCU Docker `bert_full` skeleton.
- 1-sample end-to-end run.
- Timing logs by layer/operator.

Exit criteria:

- One input text can run through both Docker paths.
- Outputs include prediction, probabilities, and end-to-end latency.

### Phase 3: Docker Full BERT Benchmark

Priority: High

Deliverables:

- 10-sample SST-2 benchmark.
- Accuracy and divergence metrics.
- Latency comparison.
- Security boundary report.

Exit criteria:

- MCU Docker is equal to or faster than CrypTen Docker, or the gap is explained by identified operators.

### Phase 4: Frontend Integration

Priority: Medium

Deliverables:

- Backend Docker orchestration endpoints.
- Frontend three-way comparison view.
- Benchmark history viewer.

Exit criteria:

- User can compare plaintext, CrypTen Docker, and MCU Docker from the UI.

## 6. Progress Checklist

### Goal 1: Operator Optimization

- [x] Cursor project progress tracking rule.
- [x] README Chinese writing rule.
- [x] Real Docker operator benchmark runner.
- [x] MCU three-role TCP path.
- [x] CrypTen two-rank Docker path.
- [x] Batch nonlinear protocols.
- [x] Communication timing split.
- [x] HP parallel receive/send.
- [x] Segmented large payload writes.
- [x] Fused CPU matmul correction and HP share backend.
- [x] BERT-base batch=1 post-optimization operator snapshot.
- [x] BERT-base batch `1,2,4` full operator ratio matrix.
- [x] Protocol/write timing split for tensor and nonlinear role binaries.
- [x] Bicoptor/wrap local-work optimization for `exp`, `sigmoid`, and `gelu`.
- [x] Standalone `exp` accepted for BERT-like batch sizes `1,2,4`.
- [x] Full post-optimization operator matrix for `large` preset.
- [x] Operator theoretical gap table for measured operators with `MCU/CrypTen > 1.5x`.
- [x] Protocol subroutine benchmark for `sign`, `wrap`, `sign-bicoptor`, and `wrap-bicoptor`.
- [x] Thread-mode batch steady-state proxy benchmark.
- [x] CUDA matmul prototype evaluated and marked not production-ready.
- [x] Clarified `rrap` as not an independent local protocol/API; Bicoptor-internal repeated truncation/randomization path is measured through Bicoptor subroutines.
- [x] Goal 1 technical optimization report.
- [ ] Investigate Bicoptor payload compression/chunked streaming for communication-heavy `exp`.
- [ ] Real persistent TCP role benchmark.
- [ ] Production CUDA/GPU matmul path after kernel and tensor-lifetime redesign.

### Goal 2: Full BERT Docker Inference

- [x] Python-level full BERT three-path comparison.
- [x] CrypTen Docker `bert_full`.
- [x] MCU Docker `bert_full` skeleton.
- [x] Persistent p0/p1/hp BERT session.
- [x] Layer-by-layer timing.
- [x] Synthetic chained hidden-share propagation inside MCU BERT session.
- [x] Real checkpoint weights and real input embedding shares in MCU BERT session.
- [x] Real Q/K/V-derived per-head attention matmul path in MCU BERT session.
- [x] Local fixed-point rescale prototype for real_io matmul outputs.
- [x] HP-clear wrap-correct numerical rescale baseline for real_io matmul outputs.
- [x] Attention/FFN bias shares exported and consumed in MCU BERT session.
- [x] HP-clear numerical feedback baseline for softmax and GeLU outputs.
- [x] Attention score scaling before softmax and real attention-mask export/consumption in local and Docker real_io runs.
- [x] HP-clear numerical LayerNorm baseline in local and Docker real_io runs.
- [x] Q/K/V bias export and consumption in MCU BERT session.
- [x] Protocol-domain-safe attention mask penalty for MCU real exponential path.
- [x] Pooler, classifier, logits, probabilities, and per-sample predictions in MCU BERT session.
- [x] End-to-end Docker benchmark over 10 SST-2 samples.
- [x] Security boundary audit for current HP-clear numerical baseline.
- [x] Warm Docker service v1: persistent containers plus backend `launch=docker_warm`.
- [x] CrypTen persistent service v2: persistent rank processes plus backend `launch=docker_service`.
- [x] MCU persistent service v2 wrapper: persistent `hp/p0/p1` wrapper processes plus backend `launch=docker_service`.
- [x] MCU service v2.1 data-preparation cache: static model shares cached and host input-share exporter kept warm.
- [x] MCU service v2.2 robustness/cache: input exporter heartbeat health check, stale-ready detection, request queue cleanup on restart, Rust-side model-share cache, and removal of the fixed 1s party sleep.
- [x] MCU service v2.3 BERT fusion: fused Q/K/V projection into one static-weight matmul plus one rescale per layer.
- [ ] Wrap-correct secure truncation/rescale after matmul.
- [ ] Secure fixed/real conversion for nonlinear feedback.
- [ ] Secure LayerNorm and pooler/classifier reveal policy.
- [ ] BERT-level fused MCU protocol calls for per-head attention score/value paths.
- [ ] True in-process MCU request loop with persistent inner TCP/session state.

### Goal 3: Frontend Integration

- [x] Existing dashboard shell.
- [x] Existing Python-level BERT mode selection.
- [x] Backend Docker orchestration API.
- [x] Frontend process/Docker launch selector for BERT sentiment mode.
- [x] Unified result schema for process and Docker BERT paths.
- [x] Minimal three-path comparison API and frontend result grid.
- [x] Basic structured role/rank log display after a Docker run.
- [x] Backend benchmark history API.
- [x] Full Docker three-path smoke test.
- [x] README and deployment docs aligned with latest Goal3 startup modes and current benchmark tables.
- [x] Original `index.html` restored as the default frontend entry; `index_new.html` kept as alternate comparison page.
- [x] Medical, finance, and sentiment scenarios support three-path comparison.
- [ ] Rich benchmark comparison charts/history picker.

## 7. Open Risks

### Risk 1: Three-Party Communication Overhead

MCU uses `p0 / p1 / hp`, while CrypTen uses two ranks. Some communication overhead is structural. The project should not hide this; instead, each operator should separate implementation overhead from protocol overhead.

Mitigation:

- Use byte/round analysis per operator.
- Use `recv_wait_s` and `recv_read_s` to separate synchronization from payload transfer.

### Risk 2: Full BERT Security Boundary

Current Python full BERT MCU path still contains plaintext components. Docker full BERT must explicitly define and reduce these plaintext boundaries.

Mitigation:

- Create a security boundary table.
- Do not claim full secure inference until all unacceptable plaintext operations are removed.

### Risk 3: Docker Startup Noise

Per-case container startup and file output can distort latency.

Mitigation:

- Add persistent process mode.
- Report cold-start and warm-run separately.
- Separate protocol time from output writing time.

### Risk 4: GPU Acceleration May Not Help Immediately

Naive CUDA kernels can be slower because of host/device copies and kernel launch overhead.

Mitigation:

- Only use GPU when tensor lifetime remains on device.
- Benchmark CPU fused backend against CUDA fused backend by shape.
- Avoid claiming GPU acceleration unless measured.

## 8. Immediate Next Actions

1. Prototype Bicoptor payload reduction for `exp`: packed/typed wire encoding, chunked HP streaming, or a narrower comparison range guarded by runtime range checks.
2. Fuse MCU BERT protocol calls: combine Q/K/V projection where possible and batch per-head attention score/value matmuls plus HP-clear conversions to reduce 362 per-request matmul calls and repeated small round trips.
3. Replace the MCU service wrapper with a true persistent TCP/session request loop so service-mode MCU does not reconnect inner p0/p1/hp sockets per request.
4. Redesign CUDA matmul around tiled kernels and persistent device tensors before re-testing GPU as a production path.
5. Replace local `--rescale-bits` truncation with wrap-correct secure truncation, then apply attention scaling/masking.
6. Replace the remaining nonlinear/LayerNorm/bias/pooler/classifier gaps with secure state flow, then run a security-aligned MCU-vs-CrypTen Docker BERT comparison.
