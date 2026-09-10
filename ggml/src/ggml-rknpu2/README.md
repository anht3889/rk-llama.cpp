# RKNPU2 backend

Rockchip RK3588 NPU backend using the RKNPU2 matmul API. The backend registers
as `GGML_BACKEND_DEVICE_TYPE_ACCEL` and uses a host buffer type, so unsupported
work falls back to the CPU backend.

## Build

Opt-in, off by default. Linux AArch64 only.

```sh
cmake -S . -B build -DGGML_RKNPU2=ON -DRKNPU2_SDK_ROOT=/path/to/runtime/Linux
cmake --build build --target llama-cli -j
```

`RKNPU2_SDK_ROOT` is the SDK `runtime/Linux` directory. Configure looks for
`$RKNPU2_SDK_ROOT/librknn_api/include/rknn_matmul_api.h` and links
`$RKNPU2_SDK_ROOT/librknn_api/aarch64/librknnrt.so` (CMake `find_library` with
`NAMES rknnrt`). Configuration fails with a clear message if the target is not
Linux AArch64, the header/library are missing, or `RKNPU2_SDK_ROOT` is unset.

AArch64 Linux on a board with the runtime installed:

```sh
# static (default) build: device list shows RKNPU2
cmake -S . -B build -DGGML_RKNPU2=ON -DRKNPU2_SDK_ROOT=/path/to/runtime/Linux
cmake --build build --target llama-cli -j 4
./build/bin/llama-cli --list-devices

# dynamic backend loading
cmake -S . -B build -DGGML_RKNPU2=ON -DRKNPU2_SDK_ROOT=/path/to/runtime/Linux \
      -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON
cmake --build build --target llama-cli -j 4
./build/bin/llama-cli --list-devices
```

STATUS: VERIFIED on a Rock 5T (RK3588) with Linux 6.1.115-vendor-rk35xx,
RKNPU driver 0.9.8, and RKNPU2 SDK 2.4.2a15 on 2026-09-09.

## Limitations

Set `ulimit -n 65536` in the shell before running a model. Multicore contexts and SDK buffers can exceed the common soft limit of 1024 open files. Descriptor exhaustion produces SDK allocation errors and invalidates performance measurements. This changes only the current shell and its child processes.

The default placement threshold is `GGML_RKNPU2_MIN_M=2`: single-token matrix operations remain on CPU. Set it to 1 to test NPU decode, or raise it to keep other small batches on CPU. It is a token-row threshold, not a model-specific cost estimator. Model-loader capability probes remain independent of this policy.

- Only 2D MUL_MAT is claimed (F32 result and F32 activations, contiguous
  constant weight, K multiple of 32, N multiple of 16, K <= 8192). Everything
  else - norms, rope, softmax, MoE `ggml_mul_mat_id`, 3D or batched matmuls,
  views, per-layer scale/div ops - runs on the CPU backend.
- Device memory queries report 0 bytes (no SDK query wired up). This does not
  block op offload: placement is decided by `supports_op`, not by memory size.
- The SDK is linked into the backend; a missing or unusable runtime surfaces as
  a loader or `rknn_*` error, never as a working NPU.
- Weights are converted to packed FP16 at admission; admission failure (bad
  budget, unsupported type, SDK error) falls back to CPU without altering the
  on-disk model or the in-memory weight bytes.
- NPU weight offload requires non-mmap loading (e.g. `llama-cli --no-mmap`).
  Under the default mmap load mode all weights stay in a CPU mmap buffer and
  every MUL_MAT runs on the CPU (results stay correct, nothing is offloaded).
  This is by design: `supports_op` never admits a weight whose buffer is a
  foreign host (mmap / CPU) buffer.

## Precision policy

The first compute path is FP16 inputs and weights with FP32 output (RK3588 FP16
matmul). A quantized weight (Q8_0, Q6_K, Q4_0, ...) that is claimed by the NPU
is dequantized to FP16 at admission, so its NPU numerics differ from the CPU
quantized kernels. Use an F16 GGUF for like-for-like CPU-vs-NPU comparison.
Existing GGUF quantizations remain usable via CPU fallback.

## Supported operations and shapes

Exactly one op offloads today: `GGML_OP_MUL_MAT`. Everything else runs on the
CPU backend.

A MUL_MAT is placed on the NPU only when all of these hold:

- 2D only: weight and activation have ne[2] == 1 and ne[3] == 1 (no batch, no
  3D, no broadcast).
- Both operands are contiguous in standard ggml layout.
- F32 activation and F32 result (a->type == GGML_TYPE_F32, dst->type ==
  GGML_TYPE_F32).
- Weight source type has a to_float conversion path: F32 directly, or F16, BF16,
  Q8_0, Q6_K, Q4_0, and the other to_float quant types. The weight is converted
  to packed FP16 at admission. A type with no to_float callback is never
  admitted.
- K (weight rows, w->ne[0]) is a multiple of 32 elements, N (weight columns and
  result columns, w->ne[1]) is a multiple of 16 elements, and K <= 8192 (the SDK
  splits B into segments above 8192, so the backend stays single-block).
- No op-hint dispatch: the Hadamard MUL_MAT variant (hint != GGML_HINT_NONE) is
  not claimed.

M (token rows, a->ne[1]) and N must be positive and fit the SDK's signed 32-bit dimensions. Empty matrices are rejected before SDK context creation. Scratch allocation remains subject to the buffer budget.

Not offloaded, and therefore run on the CPU: norms, rope, softmax, get_rows,
per-layer scale/div ops, views, 3D or batched matmuls, and MoE
`ggml_mul_mat_id`. Gemma 4 E2B is a dense model, so its MUL_MAT projections
offload. A hypothetical MoE variant keeps its expert `ggml_mul_mat_id` graphs on
the CPU.

## Fallback behavior and budget control

Placement-time denial keeps the op on the CPU silently. Denial happens when a
shape/type condition above fails, or when admission fails (budget full,
unsupported weight type, or an SDK create/alloc/sync error). The op then never
reaches the NPU and the graph runs it on the CPU backend instead.

Budget denial is the one placement denial that logs a warning:

    <func>: weight budget denied for tensor of type <type> (size <n>), CPU fallback

Shape/type denials are pure predicate checks and log nothing.

Execution-time failure returns `GGML_STATUS_FAILED`: there is no retry, and the
failed op's dst is not written (no partial result). rknn_* failures name the
call and return code (see Failure diagnostics below).

The buffer budget is `GGML_RKNPU2_BUDGET`, in bytes:

- unset: 2 GiB default (2147483648).
- 0: empty budget, every weight admission is denied, all compute runs on CPU.
- N: at most N bytes of packed weights plus per-shape A/C scratch are admitted.

The budget charges SDK-returned packed weight bytes, one shared A allocation per (M,K), and each execution context's C scratch;
device memory queries still report 0 bytes (placement is decided by
supports_op, not by the memory size query).

This is not a bound on total backend memory: SDK-internal context allocations, host cache metadata, and original host tensors are not charged. Measure process and device memory separately.

Backend instances share a buffer budget, execution contexts, and scratch buffers. A global mutex serializes execution, admission, and buffer mutation with cache invalidation. This is a conservative implementation choice for shared resources, not a hardware requirement. Contexts are retained until the last backend instance is destroyed. New shapes can lose NPU acceleration as old shapes consume the buffer budget; existing reservations are not evicted.

Eligible matrices with at least 48 output columns are partitioned into three groups aligned to 16 columns. Each group has packed weights and a context bound to one NPU core. Two persistent workers and the calling thread execute the three groups concurrently. All groups must finish and synchronize successfully before output is copied. Narrower matrices use core 0. Workers are created during placement and joined at teardown. Failed multipart admission releases only its new allocations; earlier successful reservations remain intact.

Execution contexts with the same (M,K) bind one shared FP16 normal-layout activation buffer in IOMMU domain 0. Conversion and input synchronization run once per matmul, not once per core. The buffer contents are refreshed on every operation; this is allocation reuse, not cached activation values across projections. A reference count keeps the allocation and its SDK owner context alive until the last borrowing execution entry is released, even if the allocating entry is removed first. The global mutex and worker waits prevent the next operation from overwriting input still in use. A mismatched SDK A size is rejected at placement.

## Failure diagnostics

The backend reports failures as GGML_LOG_ERROR lines prefixed with the function
name, and the budget denial as a GGML_LOG_WARN. Lines to watch for:

- `rknn_matmul_create failed ret=%d` - a matmul context (weight admission or a
  per-shape exec context) could not be created; ret is the SDK return code.
- `rknn_create_mem2 failed for %zu bytes` (and the A/C variants) - an SDK memory
  allocation failed.
- `rknn_mem_sync ... failed ret=%d` (TO_DEVICE, A, C) - a host/device memory
  synchronization failed.
- `rknn_matmul_set_io_mem A failed ret=%d` (and B, C) - binding a buffer to a
  matmul slot failed.
- `rknn_matmul_run failed ret=%d` - the NPU matmul run failed.
- `weight budget denied ... CPU fallback` - budget admission refused a weight;
  the op continues on CPU. Warning, not an error.
- `non-finite or non-FP16-representable value in weight of type %s` - a weight
  value cannot round-trip through FP16; admission fails.
- `invalid GGML_RKNPU2_BUDGET %s, using default` - the env var did not parse as
  an unsigned integer; the default budget is kept.
- `MUL_MAT weight not admissible (non-constant or ineligible)` and
  `weight admission failed for MUL_MAT` - graph compute reached a MUL_MAT whose
  weight could not be lazily admitted at execution time.

To observe placement, run with the scheduler debug env var and verbose logging:

    GGML_SCHED_DEBUG=2 ./build/bin/llama-cli ... --verbose

The scheduler prints per-node assignment lines: NPU-placed MUL_MAT nodes name
the `RKNPU2` backend, CPU-placed nodes name the CPU backend. The
`RKNPU2_HOST model buffer size` scheduler line appears on both runs and proves
weight placement only, not compute. The absence of `rknn_*` errors is not proof
of numerical correctness; compare outputs against the CPU run as in the
validation runbook. These logs do not promise any NPU speed or hardware result.

## Deferred work register (F01-F12)

Deferred items stay tracked in the approved spec and are not implemented here:
`../../../docs/superpowers/specs/2026-09-09-rknpu2-backend-design.md`. One line
each:

- F01 INT8/W8A8 matmul: efficiency follow-up after FP16 correctness; needs
  weight/dynamic-activation scales and quality fallback.
- F02 RKLLM W8A8 comparison: matched model, workload, and settings comparison
  against the RKLLM stack.
- F03 additional GGUF conversion coverage: accelerate more source types while
  keeping CPU compatibility.
- F04 multimodal validation and encoder offload: one Gemma 4 E2B image pilot and isolated vision SDK-call tracing are verified below. Broader image quality, audio, multi-image inputs, and combined language/vision budget placement remain outstanding.
- F05 smaller-memory boards: validate budgets and placement on boards with less
  RAM than the reference.
- F06 persistent packed-weight cache: cut startup conversion time after the
  in-memory layout becomes stable.
- F07 decode-specific placement and tuning: the minimum-token-row policy is implemented; per-shape crossover measurements remain deferred.
- F08 multicore scheduling and overlap: three-core column partitioning, persistent workers, and shared activation buffers are implemented. Reuse of converted values across projections and overlap between operations remain deferred. SDK combined-core mask 7 was rejected on the tested runtime.
- F09 shape padding, tiling, wider operator coverage: expand offload where
  profiling shows a benefit.
- F10 advanced quantization or mixed precision: only if F01 evidence warrants.
- F11 multiple IOMMU domains: when measured allocation limits block useful
  offload.
- F12 additional Rockchip SoCs: per-SoC validation after RK3588 is stable.

Relevant after FP16 correctness, in rough priority: F01 (INT8/W8A8), F07
(decode placement), F09 (shape/op coverage), F05 (smaller boards), F06
(persistent weight cache). If Task 6 board measurement shows decode slower than
CPU, F07 and F01 move to the top.

## Status / completion limits

Functional and performance results are reported separately, on purpose.

FUNCTIONAL: Linux AArch64 compilation and focused operator numerics were verified on a Rock 5T (RK3588, driver 0.9.8) with SDK 2.4.2a15 on 2026-09-09. After correcting alignment to K=32 and N=16 elements, 68 supported MUL_MAT cases passed against CPU across F16, BF16, Q8_0, Q6_K, and Q4_0 sources. Unsupported cases were skipped; these counts do not imply complete operator or model coverage.

The existing F16 case with m=64,n=32,k=80 reproduced an SDK rejection before the fix and is now rejected by backend eligibility. The official SDK demo also rejected N=8 with a requirement for N alignment of 16 elements.

A Gemma 3 1B Q8_0 smoke test exposed an SDK abort for a zero-row matrix during graph preparation. Rejecting empty dimensions before SDK admission resolved it. Both zero-budget CPU and default-budget runs completed the prompt "Reply with the word hello." with "Hello." using --no-mmap and a 512-token context. This is a smoke check, not a model-quality or performance benchmark.

Gemma 4 E2B BF16 also loaded and returned "hello" for that prompt with the backend enabled, --no-mmap, a 512-token context, and a 16-token generation limit. The downloaded unsloth/gemma-4-E2B-it-GGUF artifact is 9311305568 bytes, SHA-256 `1eafd61d010ce8ca09db38f370aadd64c6d792db269c365ad0d9ea2709701890`, matching the published hash at repository revision `0314792d7f1f7e229411f620751375812bb9faf2`. A verbose one-token scheduler trace confirmed RKNPU2 splits in the Gemma graph. Full-dataset quality, in-process lifecycle, and energy validation remain pending.

Reproduce the operator checks with:

```sh
./build/bin/test-backend-ops test -b RKNPU2 -o MUL_MAT -p 'type_a=f16,type_b=f32,m=64,n=32,k=80'
GGML_RKNPU2_MIN_M=1 ./build/bin/test-backend-ops test -b RKNPU2 -o MUL_MAT -p 'type_a=(f16|bf16|q8_0|q6_K|q4_0),type_b=f32'
```

The first command must report not supported without an SDK execution error. The second must execute supported cases and pass their numerical comparisons.

INITIAL SINGLE-CORE PERFORMANCE: Measured on the Rock 5T above using a separate F16 control GGUF converted from the verified BF16 artifact. Its SHA-256 is `b03d1ea2faea2ca7fd9e53c793ed055d530985bd2d52e58f67be4b21468910f8`. Both sides used the same file, `taskset -c 4-7`, four threads, no mmap, no CPU BLAS, default warmup, and five repetitions. The CPU control set `GGML_RKNPU2_BUDGET=0`; the NPU run used the default 2 GiB budget. A verbose scheduler trace confirmed RKNPU2 MUL_MAT assignments. A temporary symbol-interposition diagnostic counted direct `rknn_matmul_run` calls: zero for the budget-zero probe, 98 for the enabled `-n 1` probe, and 308 for the enabled `-n 2` probe. The additional decode step therefore submitted NPU work; scheduler split headers alone are not sufficient evidence because they include provisional planning graphs.

| Workload | CPU tok/s | NPU tok/s | NPU change |
| --- | ---: | ---: | ---: |
| pp512 | 19.568 +/- 0.394 | 20.022 +/- 0.213 | +2.32% |
| tg128 | 4.129 +/- 0.013 | 2.591 +/- 0.001 | -37.24% |

Peak process RSS was 5.34 GiB for CPU and 7.34 GiB for NPU. The prefill distributions overlap, so this run does not establish a material prefill speedup. Decode is materially slower and the speed objective is unmet. The scheduler trace shows frequent CPU/RKNPU2 alternation; decode-specific placement and INT8/W8A8 are the next performance priorities (F07 and F01).

QUALITY: A bounded one-chunk WikiText-2 raw pilot used the exact BF16 GGUF, 512-token context, batch and ubatch 512, four big CPU cores, and no warmup. CPU and NPU both reported PPL 47.7418 +/- 14.8734. Against CPU-saved logits, the NPU run had 100% same top token, maximum probability delta 0.004%, mean KL numerically zero, and maximum KL 0.000038. CPU wall time was 617.7 seconds with 5.96 GiB peak RSS; NPU wall time was 590.4 seconds with 7.82 GiB peak RSS, 4.43% lower wall time. The saved-logit format clips logits below the maximum minus 16 and quantizes log probabilities to 16 bits; its KL report reconstructed a base PPL of 45.3419, which does not match the standalone CPU PPL despite the near-zero per-token divergence. Do not use its reported PPL ratio as an accuracy claim. One chunk has a wide confidence interval and is not a full-corpus quality result.

GEMMA 3 1B F16: The `ggml-org/gemma-3-1b-it-GGUF` F16 artifact is 2006573568 bytes with SHA-256 `05bd381a5f45611ce53f4fdcc6641cf6cec68c3091d74e8a32ea591f062d3fc5`. The same performance settings as above produced:

| Workload | CPU tok/s | NPU tok/s | NPU change |
| --- | ---: | ---: | ---: |
| pp512 | 66.177 +/- 0.164 | 91.884 +/- 2.871 | +38.85% |
| tg128 | 9.989 +/- 0.014 | 4.161 +/- 0.011 | -58.35% |

Peak process RSS was 2.55 GiB for CPU and 4.77 GiB for NPU. Direct diagnostics counted zero `rknn_matmul_run` calls for budget zero, 724 for enabled `-n 1`, and 907 for enabled `-n 2`; decode therefore submitted NPU work.

The matched one-chunk F16 quality pilot reported CPU PPL 18.3269 +/- 4.6721 and NPU PPL 18.3560 +/- 4.6805, a 0.16% increase. The NPU result had mean KL 0.000139, maximum KL 0.004233, RMS probability delta 0.351%, and 99.216% same top token against the CPU-saved logits. The isolated evaluation pass took 10.49 seconds on CPU and 9.68 seconds with NPU, but total process wall time was 18.23 seconds on CPU and 35.80 seconds with NPU because the latter includes packed-weight admission and startup. This is also a one-chunk pilot, not a full-corpus quality result.

ENERGY: NOT MEASURED. No readable board power or energy sensor was available to the test user, so no power-efficiency claim is made.

### Multicore FP16 and placement experiment, 2026-09-10

The SDK rejected combined-core mask 7. Explicit aligned column partitions passed all 68 supported operator cases with `GGML_RKNPU2_MIN_M=1`. The first implementation used per-operation thread creation; it was superseded by two persistent workers. An intermediate run hit the shell's 1024-descriptor limit and was stopped. Do not use that run as performance evidence. Subsequent runs used `ulimit -n 65536`. No board-wide resource limit was changed.

Gemma 3 1B F16, same model and pp512/tg128 settings as above, five repetitions:

| Placement | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| Initial single-core NPU | 91.884 +/- 2.871 | 4.161 +/- 0.011 |
| Three-core NPU, including decode | 123.403 +/- 0.660 | 8.467 +/- 0.023 |
| Three-core prefill, CPU single-token decode | 122.635 +/- 1.054 | 9.665 +/- 0.738 |
| Final admission order, default hybrid policy | 122.755 +/- 1.066 | 10.056 +/- 0.023 |

The hybrid decode samples were 9.97345, 9.97322, 8.34613, 10.0133, and 10.0193 tok/s; the low sample is retained. This is the default policy because even improved NPU decode remained below the earlier CPU baseline of 9.989 tok/s. These results do not reproduce the reference backend's higher published throughput. Full per-shape cost calibration, shared activation storage, broader workload profiling, larger quality evaluations, and energy measurements remain follow-up work. The activation-conversion experiment below reports Gemma 3 binding and conversion costs.

The final revision reserves scratch before packing new weights and rejects weights whose minimum packed size exceeds the remaining budget. Its complete benchmark process took 147.78 seconds with 4.47 GiB peak RSS, versus 224.38 seconds and 4.77 GiB for the initial single-core backend. The isolated decode repeat before this last admission change measured 10.005 +/- 0.086 tok/s. The final one-chunk quality pilot reproduced the previous NPU PPL 18.355958, mean KL 0.000139, maximum KL 0.004233, and 99.216% same top token. That quality process took 44.57 seconds, versus 35.80 seconds initially; its evaluation pass took 16.44 seconds versus 9.68 seconds. The warm-throughput gain therefore does not apply to every workload. This remains a one-chunk quality check.

The final Gemma 4 E2B F16 control benchmark used the same board, default budget, four big CPU cores, pp512/tg128, and three repetitions. Prefill measured 34.111 +/- 0.460 tok/s and decode 4.181 +/- 0.007 tok/s, versus the initial single-core backend's 20.022 +/- 0.213 and 2.591 +/- 0.001 tok/s (five repetitions). The CPU baseline was 19.568 +/- 0.394 and 4.129 +/- 0.013 tok/s. The final process took 198.40 seconds with 7.44 GiB peak RSS; its total duration is not directly comparable to the five-repetition baseline. Gemma 4 quality was not rerun for this revision.

The final zero-budget Gemma 3 smoke benchmark (pp32/tg4, one repetition) completed with exit code 0 in 4.23 seconds. This checks successful load and execution with admission disabled, not output equivalence or nonzero low-budget behavior. Final backend source and CMake hashes matched between the local workspace and the tested board build.

Reproduce the final Gemma 3 checks (use the verified model and saved CPU logits described above):

```sh
ulimit -n 65536
taskset -c 4-7 ./build/bin/llama-bench -m gemma-3-1b-it-f16.gguf -lm none -t 4 -p 512 -n 128 -r 5 -o json
GGML_RKNPU2_MIN_M=1 ./build/bin/test-backend-ops test -b RKNPU2 -o MUL_MAT -p 'type_a=(f16|bf16|q8_0|q6_K|q4_0),type_b=f32'
taskset -c 4-7 ./build/bin/llama-perplexity -m gemma-3-1b-it-f16.gguf --no-mmap --no-warmup -f wiki.test.raw -t 4 -tb 4 -c 512 -b 512 -ub 512 --chunks 1 --kl-divergence --kl-divergence-base gemma3-f16-cpu.kld
```

### Activation conversion experiment, 2026-09-10

Matched Gemma 3 1B F16 prefill-only runs used pp512, five repetitions, four big CPU cores, non-mmap loading, and the same model as above. Increasing the budget from 2 GiB to 3 GiB did not improve throughput: 121.577 +/- 1.483 versus 120.333 +/- 3.886 tok/s. Peak RSS rose from 4.47 GiB to 5.07 GiB. The default remains 2 GiB; a larger host RAM capacity is not a reason to consume more of the SDK's 4 GiB IOMMU domain.

A temporary instrumented build measured 716 NPU matmuls across one warmup and three measured prefill passes. Of 14.508 seconds inside backend graph execution, activation conversion used 8.616 seconds (59.4%), matmul dispatch/run/wait used 5.127 seconds (35.3%), output copying used 0.484 seconds (3.3%), and binding plus explicit A/C synchronization used 0.278 seconds (1.9%). These are instrumented backend timings, not total application latency. The final observed cache held 182 weights, 1,395,523,584 charged weight bytes and 56,492,032 charged scratch bytes. Binding was not the dominant cost.

The retained change uses AArch64 NEON conversion for activation rows, with GGML's signed canonical NaN result and a scalar tail. It leaves weight packing, CPU decode placement, memory ownership, synchronization, and the budget unchanged. A temporary test compared 1,304,570 values bit-for-bit against the generic GGML converter: all FP16 values expanded to F32, finite FP16 midpoints and their adjacent F32 values, and deterministic random F32 bit patterns. It also checked lengths 0 through 33 and unaligned output with sentinels. The conversion-speed check failed before the change (ratio 0.999) and passed afterward (12.550 ms versus 62.492 ms for 20 conversions of 512 x 1536 values, ratio 0.201). All 68 supported MUL_MAT cases passed afterward.

Without timing instrumentation, matched prefill throughput increased to 217.153 +/- 1.624 tok/s, 78.6% above the 121.577 tok/s control. The process took 41.85 seconds versus 53.36 seconds, with unchanged 4.47 GiB peak RSS. The one-chunk quality pilot retained PPL 18.355958, mean KL 0.000139, maximum KL 0.004233, and 99.216% same top token. That quality process took 43.65 seconds, with a 14.37-second evaluation pass; it is not a full-dataset accuracy result.

Reproduce the isolated throughput comparison with `taskset -c 4-7 ./build/bin/llama-bench -m gemma-3-1b-it-f16.gguf -lm none -t 4 -p 512 -n 0 -r 5 -o json`, after `ulimit -n 65536`. Set `GGML_RKNPU2_BUDGET=2147483648` or `3221225472` for the budget comparison. On the validation board, `build/comparison/` retains `prefill-2g-before`, `prefill-3g-before`, `prefill-profile-verbose`, `prefill-neon`, and `ppl-gemma3-neon` logs and resource records, plus `rknpu-convert-check.cpp`. Timing instrumentation is not part of the production source.

A separate five-repetition pp512/tg128 confirmation measured 215.933 +/- 1.943 tok/s prefill and 10.085 +/- 0.016 tok/s decode (`bench-gemma3-neon-final`). CPU decode is retained; the small difference from the previous 10.056 tok/s decode result is not claimed as an NPU improvement.

The Gemma 4 E2B F16 control prefill check measured 39.699 +/- 0.672 tok/s (pp512, three repetitions), versus the prior revision's 34.111 +/- 0.460 tok/s prefill measurement. The new run was prefill-only (`prefill-gemma4-neon`), took 77.01 seconds and peaked at 7.44 GiB RSS. Gemma 4 decode and quality were not rerun for this conversion-only change.

At this stage, shared activation storage, conversion reuse across projections, and microbatch tuning remained follow-ups. The subsequent experiments below evaluate them independently; no shared-buffer lifetime or synchronization change was included in the conversion-only revision.

### Microbatch sweep after NEON conversion, 2026-09-10

Gemma 3 1B F16, unchanged NEON backend, default 2 GiB budget, four big CPU cores, logical batch 2048, three repetitions with warmup, and a fresh process for every case:

| Microbatch rows | pp512 tok/s | pp1024 tok/s |
| ---: | ---: | ---: |
| 128 | 187.572 +/- 0.203 | 170.794 +/- 0.667 |
| 256 | 218.500 +/- 0.387 | 207.290 +/- 0.198 |
| 512 | 216.821 +/- 0.587 | 210.137 +/- 0.133 |
| 1024 | Not exercised | 209.597 +/- 0.117 |

Keep microbatch 512 as the baseline. The 256-row setting was 0.8% faster for pp512 but 1.4% slower for pp1024; it also had higher observed peak RSS and longer process duration in both comparisons. The 1024-row setting did not improve pp1024, while 128 rows was clearly slower. The pp512/ub1024 case was omitted because a 512-token prompt cannot exercise 1024 activation rows. Compare configurations within each prompt-length column, not across different prompt lengths.

All seven processes exited successfully, with no SDK failure messages found in their verbose logs. These are throughput measurements, not additional quality or energy checks. No clock or governor was changed; a read-only sample during the sweep reported the NPU at its configured maximum of 1 GHz and DMC at 2.4 GHz. This sample does not establish fixed clocks throughout every run.

Reproduce each case in a separate process after `ulimit -n 65536`: `GGML_RKNPU2_BUDGET=2147483648 taskset -c 4-7 ./build/bin/llama-bench -m gemma-3-1b-it-f16.gguf -lm none -t 4 -b 2048 -ub 512 -p 1024 -n 0 -r 3 -o json --progress -v`, substituting the microbatch and prompt sizes above. Logs and resource records are retained on the board under `build/comparison/sweep-g3-p<PROMPT>-u<MICROBATCH>.*`.

The refreshed NEON profile used pp512/ub512 and three measured passes, excluding warmup from the following breakdown. It recorded 537 matmuls, with an average 1,742.943 ms inside backend graph execution per pass:

| Stage | Mean ms per pass | Share of backend time |
| --- | ---: | ---: |
| Matmul dispatch, run and wait | 1245.020 | 71.4% |
| Activation conversion | 316.153 | 18.1% |
| Output copy | 112.307 | 6.4% |
| Context lookup and B binding | 29.954 | 1.7% |
| Input synchronization | 23.977 | 1.4% |
| Output synchronization | 14.867 | 0.9% |

The instrumented end-to-end mean was 2,366.519 ms per prefill pass (216.353 +/- 0.629 tok/s). The difference from backend time includes CPU work and scheduling; it is not further attributed by this instrumentation. The cached weight and scratch totals matched the previous profile. Across warmup plus measured passes, conversion fell from the earlier 8.616 seconds to 1.275 seconds, while matmul dispatch/run/wait remained near 5.1 seconds.

This profile identified shared activations as the next candidate. Eliminating two-thirds of the measured input conversion and synchronization would save about 227 ms per pass: an optimistic 10.6% throughput increase at this profile's end-to-end latency, before any added binding or lifetime-management overhead. This was an analytical bound for that specific reduction, not a measured speedup or an upper bound on all possible optimizations. The implementation and validation follow below. Reuse of converted values across separate projections is a distinct follow-up with additional invalidation requirements.

The temporary source and logs are retained on the board as `build/comparison/rknpu-neon-profile.cpp` and `neon-profile-current.*`. Production source was restored to SHA256 `cf43c7baa486bd15e269a70f82208d0c816b1664578a26327e5e4126b8edbe88`, rebuilt, and passed all 68 supported operator tests (`neon-profile-restore-test.log`). This sweep/profile round changes no production code or defaults.

### Shared activation implementation, 2026-09-10

The retained implementation shares an A allocation across cached execution contexts with the same (M,K). It binds A during placement, charges the allocation once, and converts/synchronizes once per matmul. Output buffers, worker scheduling, FP16 precision, microbatch 512, the 2 GiB budget, and CPU single-token decode policy are unchanged.

Matched five-repetition Gemma 3 1B F16 pp512/ub512 prefill-only runs measured 218.832 +/- 2.601 tok/s before and 246.788 +/- 0.414 tok/s after, a 12.8% improvement. Peak RSS fell from 4,687,316 KiB to 4,658,776 KiB (27.9 MiB lower); process duration fell from 41.58 to 40.17 seconds. A separate pp512/tg128 confirmation measured 247.615 +/- 0.315 tok/s prefill. The one-chunk quality pilot retained PPL 18.355958, mean KL 0.000139, maximum KL 0.004233, and 99.216% same top token; its process duration was 43.05 seconds. Full-dataset quality and energy remain unverified.

All 68 supported MUL_MAT cases passed. The temporary `build/comparison/rknpu-shared-check.cpp` test first failed on core 1 against the previous backend (0 instead of the expected 64 after one input write), then passed with sharing. It verifies cross-context A numerics, exact single-charge scratch accounting, budget denial with a new or existing activation shape, release/recreation of the allocating execution entry while borrowers remain, refreshing input through the retained SDK owner, alternating shapes over 12 teardown cycles, partial weight-admission rollback, preservation of earlier reservations, and zero-budget placement denial. Expected budget-denial warnings are retained in its log.

AddressSanitizer subsequently caught a teardown use-after-free in the new release path: teardown passed the map entry's key by reference, erased that entry, then read the key to remove the shared activation. Passing the key by value fixes its lifetime. The expanded test then passed under AddressSanitizer, with stable descriptor counts across 12 cycles (`shared-lifecycle-asan-fixed.log`). LeakSanitizer was disabled; the SDK and linked GGML libraries were not sanitizer-instrumented, so this is not a complete device-memory leak audit. The initial performance/quality records above predate this teardown correction; final revalidation is recorded separately below.

Board evidence is retained under `build/comparison/shared-before.*`, `shared-after.*`, `shared-confirm.*`, `shared-quality.*`, `shared-test.log`, and `shared-lifecycle.log`. Reproduce throughput using the same commands as the preceding NEON experiment. This change does not reuse converted values across different matmuls or overlap operations.

Final verification after the teardown correction used backend SHA256 `09bf47094507b3b6dfbc0b54e412edb5f226100db2b84cf6e40e5f43f2db7a2e`, matching local and board source:

- Gemma 3 pp512/tg128, five repetitions: 247.909 +/- 0.444 tok/s prefill and 10.080 +/- 0.016 tok/s decode. The pp512 result is 13.3% above the fresh 218.832 tok/s pre-change control. Process duration was 124.22 seconds, including both benchmark cases, with 4,658,772 KiB peak RSS.
- Gemma 3 quality pilot: PPL 18.355958, mean KL 0.000139, maximum KL 0.004233, and 99.216% same top token. Process duration was 42.61 seconds; the evaluation pass took 14.23 seconds. This remains a one-chunk check.
- Gemma 4 E2B F16 prefill cross-check, pp512 with three repetitions: 41.297 +/- 0.137 tok/s, compared with the earlier NEON-only run's 39.699 +/- 0.672 tok/s. Process duration was 75.38 seconds and peak RSS was 7,802,772 KiB. Gemma 4 decode and quality were not rerun for this change.
- Zero-budget Gemma 3 model execution (pp32/tg4, one repetition) completed successfully in 4.38 seconds. This smoke check does not establish token-by-token equivalence to a separate CPU run.
- All 68 supported operator tests passed again. The AddressSanitizer lifetime test passed with stable descriptor counts of 7 at its post-teardown sampling points.

Final logs use `build/comparison/shared-fixed-confirm.*`, `shared-fixed-quality.*`, `shared-fixed-gemma4.*`, `shared-fixed-zero.*`, `shared-fixed-test.log`, and `shared-lifecycle-asan-fixed.log`. The implementation is retained based on the repeated throughput gain and these correctness checks, with broader quality, energy, and device-memory lifecycle testing still outstanding.

### Gemma 4 E2B image pilot, 2026-09-10

The unchanged backend with SHA256 `09bf47094507b3b6dfbc0b54e412edb5f226100db2b84cf6e40e5f43f2db7a2e` was tested with `llama-mtmd-cli` on the same RK3588 board. The language model is the F16 control above. The matching [Unsloth F16 projector](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/blob/main/mmproj-F16.gguf) has SHA256 `140be8d7849741f88c50757d529b84373ee8e27052cc2236855b537f4a8215fa`, matching its published hash. The input is `tools/mtmd/test-1.jpeg`, SHA256 `2dff664c0c8aaea18aff8cbe7e868845b775e90cdd7a0bac98df709b131deaa3`. Its 624 x 480 preprocessed image produced 130 image tokens.

The prompt was `Read the largest headline in this image. Answer with only the headline.` All answer runs used four big CPU cores (`taskset -c 4-7`), `-t 4 -tb 4 -c 2048 -b 512 -ub 512 -n 256 --temp 0 --seed 1234 --jinja -fa off -ngl 999 -v --no-warmup`. These are single fresh-process observations, not repeated throughput measurements. Initial 32-token runs stopped inside reasoning text and do not count as complete-answer quality checks.

With non-mmap loading, the budget-zero CPU run and both default-budget runs (CPU vision or explicit `--mmproj-device RKNPU2`) returned byte-identical generated text, ending with `MEN WALK ON MOON`. The CPU process took 68.92 seconds with 6,543,164 KiB peak RSS; CPU vision with NPU-enabled language-model placement took 88.82 seconds with 8,660,796 KiB; enabling the projector backend too took 90.38 seconds with 8,662,128 KiB. Image encoding took 6.600, 6.590, and 6.641 seconds respectively. Decode was approximately 4.1 tokens/s in all three cases. This short workload did not benefit from the NPU-enabled configuration.

Separate one-generated-token diagnostics used a temporary `LD_PRELOAD` wrapper around `rknn_matmul_run` and `GGML_SCHED_DEBUG=2`. Both the budget-zero control and the non-mmap default-budget language-plus-vision run made zero SDK matmul calls. Thus the non-mmap image pilot establishes CPU fallback compatibility, not NPU numerical accuracy. Provisional scheduler splits and `CLIP using RKNPU2` alone are not evidence of executed NPU work.

For isolated vision, mmap excludes language-model weights from NPU admission while `--mmproj-device RKNPU2` gives the projector its own backend host buffer. This diagnostic made 339 successful SDK matmul calls, all during image encoding, and the vision graph had 227 splits. The language-model graph had one CPU split. Traced timings are not used as performance measurements.

The uninstrumented mmap pair also returned the correct final headline. Unlike the all-fallback pair, its reasoning text differed, so it is not token-by-token equivalent. Both processes exited zero:

| Measurement | CPU, zero budget | NPU vision, CPU language model |
| --- | --- | --- |
| Cold image encoding | 6.627 s | 6.578 s |
| Image embedding evaluation in language model | 6.715 s | 6.753 s |
| Complete process | 65.97 s | 72.67 s |
| Peak RSS | 5,718,072 KiB | 6,094,268 KiB |
| Autoregressive eval calls | 182 | 209 |

The total times include different generated lengths and must not be interpreted as a throughput regression. Cold image encoding includes graph preparation and first-use admission; the 0.049-second difference from one pair does not establish a speedup. Repeated warm encoder measurements and numerical embedding comparisons are recorded below. These results are retained as `mm-validated-0-cpu-mmap.*` and `mm-validated-0-vision-mmap.*`.

Reproduce the isolated-vision pair with the verified files, substituting local paths:

```sh
ulimit -n 65536
GGML_RKNPU2_BUDGET=0 taskset -c 4-7 ./build/bin/llama-mtmd-cli -m /path/gemma-4-E2B-it-F16-control.gguf --mmproj /path/gemma-4-E2B-it-mmproj-F16.gguf --image tools/mtmd/test-1.jpeg -p 'Read the largest headline in this image. Answer with only the headline.' --load-mode mmap --no-warmup -t 4 -tb 4 -c 2048 -b 512 -ub 512 -n 256 --temp 0 --seed 1234 --jinja -fa off -ngl 999 -v --device none --no-mmproj-offload
taskset -c 4-7 ./build/bin/llama-mtmd-cli -m /path/gemma-4-E2B-it-F16-control.gguf --mmproj /path/gemma-4-E2B-it-mmproj-F16.gguf --image tools/mtmd/test-1.jpeg -p 'Read the largest headline in this image. Answer with only the headline.' --load-mode mmap --no-warmup -t 4 -tb 4 -c 2048 -b 512 -ub 512 -n 256 --temp 0 --seed 1234 --jinja -fa off -ngl 999 -v --device none --mmproj-device RKNPU2
```

Control pitfalls: `--device none` does not exclude ACCEL buffer types from the language-model CPU buffer candidate list. An `-ot '.*=CPU'` override also considers those candidates and did not isolate CPU weights. Use `GGML_RKNPU2_BUDGET=0` for a no-NPU control. Keep the failed-control logs, but do not label their timings CPU-only. The follow-up below identifies budget exhaustion in the combined configuration.

Board artifacts are in `build/comparison/`: `mm-validated-0-cpu.*`, `mm-answer-hybrid.*`, `mm-answer-npu.*`, `mm-validated-1-cpu.*`, `mm-validated-1-npu.*`, and `mm-validated-1-vision-mmap.*`. Commands are retained in `rknpu-mm-validated.sh`, and each resource JSON records its command, exit status, wall time, and peak RSS. All listed completed runs exited zero. The earlier `mm-image-*`, `mm-answer-cpu.*`, and `mm-validated-1-vision.*` records contain the short-generation or failed-isolation controls described above.

This is one OCR-style image smoke test, not a multimodal accuracy benchmark. Audio, video, multiple images, broader embedding/logit error comparisons, and energy remain unverified. No backend or multimodal implementation changes were made for this pilot.

### Combined admission diagnosis and warm vision follow-up, 2026-09-10

A read-only GDB run of the combined non-mmap configuration stopped at `mtmd_batch_encode`. The shared context contained a 2,147,483,648-byte budget, 2,050,228,224 bytes of persistent weights (1,955.25 MiB), and 97,255,424 bytes of cached scratch (92.75 MiB): exactly zero bytes remained. After image encoding started, the first execution-context request observed was M=130, K=1536, partition N=688, core 0, with the same full counters. The existing admission checks therefore cannot allocate scratch for the actual image-embedding evaluation shape or admit new vision weights. Packed weights and planning-shape scratch remain resident until teardown; this is allocation starvation, not an unsupported vision matmul kernel. `mm-budget-gdb.log` and `rknpu-mm-budget.{gdb,sh}` retain the diagnostic; its timing is not a benchmark.

An isolated encoder probe loaded the same model with mmap but did not create a language-model execution context. It selected CPU vision first, then NPU vision, and encoded the same preprocessed image six times per context. The first call is cold; the following five calls are the warm sample. No SDK tracing or debugger was active during this measurement. Both modes used four big cores, four threads, and disabled flash attention.

| Encoder | Cold call | Five warm calls, mean +/- sample standard deviation |
| --- | --- | --- |
| CPU | 6.591 s | 6.605 +/- 0.008 s |
| NPU with CPU fallback | 6.703 s | 3.412 +/- 0.003 s |

The measured warm encoder speedup is 1.94x for this one image. This does not include language-model prefill/decode, image preprocessing, model loading, or end-to-end request latency. The two backend groups ran sequentially in one process, so broader workload and order-reversed measurements remain useful before generalizing the result.

All 199,680 output embedding values were finite. All six outputs were exactly repeatable within each backend. Comparing CPU and NPU yielded RMSE 0.007520776, relative L2 error 0.009872480, cosine similarity 0.999951270, and maximum absolute error 0.161024809. These are measured errors, not an established model-quality acceptance threshold; the earlier complete-answer comparison passed for this image but does not replace broader evaluation.

The temporary probe and successful results are `build/comparison/rknpu-mm-warm.cpp` and `mm-warm-fixed.{stdout,stderr}`. It uses the existing `mtmd_encode_chunk` API and excludes marker text chunks from encoding. Initial probe attempts incorrectly assumed a single tokenizer chunk and then used vocabulary-only loading, which supplied no embedding dimensions; those attempts are not numerical evidence. The corrected probe requires a nonempty embedding and exited zero.

Next: design admission headroom for actual execution shapes while preserving the placement-time CPU fallback guarantee and previously admitted reservations. Combined language/vision weight prioritization also needs an explicit policy. Increasing the overall budget alone does not address the greedy-admission failure mode. No production behavior was changed in this investigation.

## Hardware validation runbook (Gemma text)

STATUS: PARTIALLY VERIFIED. Build, device registration, operator numerics, Gemma text smoke tests, scheduler placement, matched F16 throughput, and a one-chunk BF16 quality pilot were run on the RK3588 board described above. Low-budget variants beyond zero, in-process lifecycle, full-dataset quality, and energy remain to be run.

Targets used below: `llama-cli`, `llama-perplexity` (build both). All logs keep
the default ASCII format.

### 1. Build on the board

```sh
cmake -B build -DGGML_RKNPU2=ON -DRKNPU2_SDK_ROOT=/path/to/runtime/Linux
cmake --build build --target llama-cli llama-perplexity -j 4
```

Expected successful configure lines include:

```text
RKNPU2 SDK root:    /path/to/runtime/Linux
RKNPU2 include dir: /path/to/runtime/Linux/librknn_api/include
RKNPU2 library:     /path/to/runtime/Linux/librknn_api/aarch64/librknnrt.so
```

Confirm the device is registered:

```sh
./build/bin/llama-cli --list-devices
```

Look for a `RKNPU2` row. It will show 0 bytes of memory; that is expected and
does not block offload (placement is decided by supports_op, not memory size).

### 2. Discover the exact runtime flags

Do NOT trust flags written in this document for offload or perplexity options;
read the binary's own help first and use what it prints:

```sh
./build/bin/llama-cli --help
./build/bin/llama-perplexity --help
```

Record the exact names of: the offload/device arguments (if any), the verbose
logging flag, and the perplexity divergence/KL comparison options. The RKNPU2
device is an ACCEL type, so per-layer `--n-gpu-layers`-style GPU flags do not
target it; offload is automatic per-op via supports_op.

### 3. CPU-only vs NPU for the same prompt

Control NPU use with the budget env var (this is the reliable switch):

```sh
# CPU only
GGML_RKNPU2_BUDGET=0 GGML_SCHED_DEBUG=2 ./build/bin/llama-cli \
    -m gemma-e2b.gguf -p "..." -n 64 --verbose --no-mmap

# NPU enabled (default budget)
GGML_SCHED_DEBUG=2 ./build/bin/llama-cli -m gemma-e2b.gguf -p "..." -n 64 --verbose --no-mmap
```

Use the SAME prompt and the SAME GGUF twice. Weight offload requires
`--no-mmap`: with default mmap the NPU-enabled run silently does everything on
the CPU. With `--verbose` plus `GGML_SCHED_DEBUG=2` the scheduler prints
per-node assignment lines. Confirm:

- With `--no-mmap`, the `RKNPU2_HOST model buffer size = ... MiB` line appears
  in both runs. This proves weight PLACEMENT only, not compute; placement
  happens even when the budget is 0. Under default mmap no weight is placed in
  RKNPU2_HOST, so the line is absent from both runs.
- In the NPU run, MUL_MAT nodes are assigned to the `RKNPU2` backend (node
  assignment lines name `RKNPU2`). In the budget-0 run they name the CPU
  backend.
- Prefill (large M) can use NPU; decode (M=1) remains on CPU by default. Repeat with `GGML_RKNPU2_MIN_M=1` to exercise NPU decode. Decode is every generated token after prompt eval.
- No `rknn_matmul_create failed`, `rknn_mem_sync ... failed`, or
  `rknn_matmul_run failed` lines appear in the NPU run.

If the output tokens differ between the two runs, that is expected when weights
were dequantized to FP16 for the NPU. To make them comparable, repeat with an
F16 GGUF.

### 4. Low budget

```sh
GGML_RKNPU2_BUDGET=0        ./build/bin/llama-cli -m gemma-e2b.gguf -p "..." -n 64 --no-mmap
GGML_RKNPU2_BUDGET=67108864 ./build/bin/llama-cli -m gemma-e2b.gguf -p "..." -n 64 --no-mmap
```

("67108864" is 64 MiB.) Confirm:

- Model load succeeds in both cases (a zero budget must never abort load).
- Output is identical to the all-CPU run; no error lines; with `--verbose` and
  `GGML_SCHED_DEBUG=2` the MUL_MAT nodes fall back to the CPU backend under a
  0 budget.
- The GGUF and the in-memory weight bytes are unchanged: admission only reads
  the host weight buffer and writes a separate packed FP16 buffer, never the
  model bytes. Re-run the CPU-only command afterward and confirm bit-identical
  output to the earlier CPU-only run.

### 5. Repeated shapes and resource counts

```sh
./build/bin/llama-cli -m gemma-e2b.gguf -p "<long prompt>" -n 256 -c 4096 -v --no-mmap
```

A single run alternates prefill (large M) and decode (M=1) naturally. Record:

- `rknn` SDK allocation/driver lines, if any are printed.
- Peak resident memory (e.g. `/usr/bin/time -v` RUsage peak, or watch
  `/proc/<pid>/status` VmRSS).
- That execution contexts are created at placement-time admission, charged to
  scratch, and cached per (M,K,N) shape. Contexts are released when the backend
  last live backend instance is destroyed (teardown); there is no eviction during a graph's lifetime. The buffer budget bounds charged scratch bytes, not SDK-internal context memory.

On RK3588, exercise prompt lengths 32, 64, 128, 256, 512, and 1024 with decode between prompts in one process retaining the same backend instances. Repeat the sequence, then revisit the initial lengths. Use an existing runner that preserves those instances; confirm its lifecycle before interpreting the results. Separate process runs cannot validate cache retention.

Record actual (M,K,N) shapes, context count, charged packed-weight and scratch bytes, process RSS, available driver memory counters, CPU/NPU placement, and latency after each request. Repeat under a smaller buffer budget. Previously admitted contexts must remain usable; new shapes denied at placement must run on CPU. Repeated shapes must reuse their contexts without increasing charged scratch. Report any SDK-internal memory growth separately. After the last backend instance closes, verify that SDK contexts and buffers are released. These hardware measurements remain documented-not-verified.

True in-process multi-model load/unload is not exercised by the released
binaries; note that limitation in the sign-off. A loop of separate `llama-cli`
runs only exercises process-level load/unload, not the in-process weight and
context lifetime paths.

### 6. Perplexity comparison

```sh
# CPU only
GGML_RKNPU2_BUDGET=0 ./build/bin/llama-perplexity -m gemma-e2b.gguf -f wiki.test.raw --no-mmap

# NPU enabled
./build/bin/llama-perplexity -m gemma-e2b.gguf -f wiki.test.raw --no-mmap
```

Use the SAME dataset and GGUF. First check `llama-perplexity --help` and use
its current KL/logit comparison options to compare the two logit files. Record:

- The all-CPU perplexity and the NPU perplexity.
- A separate FP32 (or F16) baseline run when a floating-point GGUF is
  available, so dequant-to-FP16 conversion is not conflated with NPU
  arithmetic.
- Any `rknn_*` errors in the NPU run (there must be none).

### 7. Task-4 carry items to verify on the board

- Cross-context B binding: confirm each MUL_MAT binds a weight mem created by a
  weight-scoped rknn context into a shape-scoped exec context
  (`rknn_matmul_set_io_mem(exec_ctx, weight_mem, &io.B)`), and results are
  correct across distinct (M,K,N) shapes.
- `io_attr.B.size` coherence: the SDK-returned packed B size is non-zero and >=
  the native-layout packed bytes before the weight mem is allocated.
- Exec-context cache: with the full Gemma text graph, per-shape contexts are
  created at placement, cached, and released at teardown; the scratch budget
  bound holds and no eviction happens during the graph's lifetime.

Sign-off: for every item above, write either "pass" with the recorded numbers,
or the exact failing command and log snippet. Mark the whole section
DOCUMENTED-NOT-VERIFIED until a board run signs it off.

## Performance measurement runbook (Task 6)

STATUS: PARTIALLY VERIFIED. The matched pp512/tg128 comparison and bounded quality pilot are recorded above. Longer contexts, energy, and the remaining runbook cases are not verified. Run the text-validation runbook first with the same build, GGUF, SDK, and driver; do not skip correctness to chase speed.

### 8. llama-bench: confirm arguments, then matched CPU vs NPU

Confirm the real flag names before trusting anything written here:

```sh
./build/bin/llama-bench --help
```

The RKNPU2 device is ACCEL type, so classic GPU-layer flags do not target it;
offload is automatic per-op. The reliable CPU/NPU switch is `GGML_RKNPU2_BUDGET`
(0 = all CPU, unset = default budget). Standard workloads used below: pp512
(prompt processing, `-p 512`) and tg128 (text generation, `-n 128`). Run each
side with several repetitions and identical threads and context:

```sh
# CPU - prompt processing and text generation
GGML_RKNPU2_BUDGET=0 ./build/bin/llama-bench -m gemma-e2b.gguf -p 512 -n 1  -t 4 -r 5 --no-mmap <flags from --help>
GGML_RKNPU2_BUDGET=0 ./build/bin/llama-bench -m gemma-e2b.gguf -p 0   -n 128 -t 4 -r 5 --no-mmap <flags from --help>

# NPU - same flags, default budget
./build/bin/llama-bench -m gemma-e2b.gguf -p 512 -n 1  -t 4 -r 5 --no-mmap <flags from --help>
./build/bin/llama-bench -m gemma-e2b.gguf -p 0   -n 128 -t 4 -r 5 --no-mmap <flags from --help>
```

Keep `-t` (threads), the context-size flag, and batch settings IDENTICAL on both
sides of each pair. Confirm with `GGML_SCHED_DEBUG=2` that the NPU run places
the MUL_MAT nodes on RKNPU2 and the budget-0 run places them on CPU. Confirm
the exact no-mmap/load-mode flag name from `llama-bench --help`; the NPU side
must not silently run all-CPU because weights were mmap-loaded.

Then one longer representative context for the exact model (within its context
limit), CPU vs NPU:

```sh
GGML_RKNPU2_BUDGET=0 ./build/bin/llama-bench -m gemma-e2b.gguf -p 4096 -n 1 -t 4 -r 5 --no-mmap <flags from --help>
./build/bin/llama-bench -m gemma-e2b.gguf -p 4096 -n 1 -t 4 -r 5 --no-mmap <flags from --help>
```

### 9. Metadata to record for every run

- model file: sha256sum of the GGUF, its source quantization (F16, Q8_0, ...),
  and the fact that NPU weights were dequantized to FP16 at admission.
- CPU settings: thread count, context size, batch size, and CPU BLAS/OpenMP
  configuration of the build.
- SDK and driver: librknnrt library version (if printable), kernel driver
  version, kernel version, and the exact RKNPU2 SDK path used to build.
- NPU clocks and temperatures: NPU frequency and board thermal sensors before
  and after each run (e.g. /sys/class/devfreq and thermal zones).
- startup conversion time: wall time from process start to first token; this
  includes the dequant-to-FP16 packing of every admitted weight (F06 target).
- warm throughput: warm up first, then steady-state tokens/second for prefill
  and decode separately.
- peak memory: process peak resident set (e.g. /usr/bin/time -v or
  /proc/<pid>/status VmHWM) and the packed-weight bytes charged against the
  budget.

### 10. Energy over the same token workload

Measure time and board energy over the SAME workload as above, and disclose
boundaries and idle subtraction. State the method, not a number:

- fix a token workload and sample whole-board power during it (external meter
  or on-board PMIC/INA sensors);
- measure idle power over an equal window with the process stopped;
- energy = run-window joules minus idle-baseline joules for the same duration;
- report joules/token = energy / tokens and average watts = energy / seconds,
  plus exact window boundaries and sample rate;
- disclose that on-board sensors see the whole board, not the NPU alone.

Record CPU-run energy the same way; the CPU-vs-NPU delta is the claim, not an
absolute NPU number.

### 11. Phase comparison and the success criterion

Report prefill and decode SEPARATELY; never blend them. If decode on the NPU is
slower than decode on CPU, mark the speed objective UNMET in the sign-off and
promote F07 (decode-specific placement) and F01 (INT8/W8A8). Do not bend the
success criterion or re-report a single blended number.
