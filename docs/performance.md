# Performance and statistical-arbitrage audit

This audit was run on 2 August 2026 to answer two questions:

1. Which low-latency patterns in the Imperial HFT reference implementation
   transfer cleanly to Hypercube?
2. Does Hypercube's pairs example preserve the useful rolling-window idea
   while improving its statistical and systems boundaries?

The answer is selective rather than wholesale. Reuse, bounded rings,
configuration-time planning, single-pass projection, and algorithmic top-k
selection transferred well. Explicit prefetching and unconditional
architecture-specific SIMD did not earn a place in the portable path.

## Reference and method

The local reference was
[`0burak/imperial_hft`](https://github.com/0burak/imperial_hft) at commit
[`5783606`](https://github.com/0burak/imperial_hft/commit/578360605e1267628e18ea51677af2beef3cbf2e).
Its checked-in Google Benchmark binaries were run with short repeated samples:

```bash
./mybenchmark \
  --benchmark_min_time=0.05s \
  --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true
```

Those measurements are an audit aid, not Hypercube results. Several reference
microbenchmarks isolate one operation, use different compiler flags, or do not
model contention. Their value is in forming hypotheses that Hypercube then
tests in its own workload.

## Pattern cross-check

| Reference pattern | Local reference result | Hypercube decision |
| --- | ---: | --- |
| Buffer reuse / circular window | Pairs median 334.7 µs to 172.9 µs, 1.94× | Adopted as preallocated publication buffers and `RollingMoments`. |
| Combined fixed buffer, one pass, and AVX | 334.7 µs to 66.3 µs, 5.05× | Adopt the algorithmic and reuse parts; keep SIMD portable and measured separately. |
| Configuration-time dispatch | Tiny microbenchmark favored static dispatch | Adopted at the appropriate boundary: stable node graphs compile once and reuse an indexed topological plan. |
| Branch reduction | 4.26 ns to 2.89 ns, 1.47× | Retained required-input early exit and removed repeated dependency-readiness scans from each generation. |
| Loop unrolling | 215 ns to 58 ns in its isolated loop | Left to LLVM until a Hypercube profile identifies a specific loop and a portable benchmark proves the change. |
| Explicit SIMD | Array add 11.9 µs to 7.5 µs; pairs-only SIMD 302.9 µs | Deferred. Stable Rust portable SIMD remains experimental, and the pairs result shows that SIMD is secondary to the algorithm. |
| Software prefetch | 5.640 ms to 5.632 ms, no material gain | Rejected for current sequential vectors; hardware prefetch and contiguous scans already fit this access pattern. |
| Forced inlining | Pairs 334.7 µs to 276.2 µs, 1.21× | No blanket `inline(always)`. Release optimization can inline small Rust functions; annotations require a named benchmark. |
| Slow-path outlining | 4.28 µs to 6.29 µs, 47% slower | Rejected as a general rule. Durability is separated semantically from visibility instead of relying on code layout. |
| Atomic versus mutex counter | 16.6 µs versus 121.7 µs in a single-operation test | Not generalized. Hypercube already uses sequence counters and a bounded Disruptor where their concurrency contracts apply. |
| Full sort for a small top set | Not isolated in the reference | Replaced with linear selection plus a sort of only the selected prefix. |

This is deliberately a data-oriented design, not a collection of compiler
incantations. The [Rust Performance Book](https://nnethercote.github.io/perf-book/)
also recommends starting with algorithms and data structures, reducing hot
allocations, and benchmarking any lower-level change.

## Changes made

### Reusable execution plans

An `Update` still carries its complete, replayable node declaration. The
engine now validates and topologically compiles a changed declaration into
indexed dependency edges. Later generations with an identical declaration
reuse that plan. `HypercubeEngine::graph_compilations()` exposes the number of
plans built, making accidental per-frame topology churn observable.

The values are still recalculated for every coherent generation. This is plan
reuse, not incremental value caching, so generation and observation-time
semantics are unchanged.

### One-pass publication and explicit durability

`SlicePublisher` now:

- caches entity-to-slot resolution;
- allocates one vector per configured node at construction;
- clears and fills those vectors in one pass over snapshot cells; and
- updates every writer epoch before applying a chosen flush policy.

Three policies are explicit:

| Policy | Return condition | Intended use |
| --- | --- | --- |
| `MemoryMapped` | New even epochs and payloads are visible through the mapping. | Live views whose recovery truth is elsewhere. |
| `Async` | The operating system accepted asynchronous flush requests. | Best-effort background persistence. |
| `Durable` | Every mapping reports its dirty pages durably stored. | A synchronous durability boundary. |

`SlicePublisher::publish()` remains `Durable` for compatibility. The synthetic
server selects `MemoryMapped` explicitly. Visibility is not durability, and no
policy makes independently published slices one atomic multi-slice generation.
The distinction follows the `memmap2` contract: `flush()` waits for durable
storage, while `flush_async()` only initiates the operation.

### Constant-time rolling moments

For pair (j), Hypercube continues to generate the known cointegrating
residual

```text
y_j,t = log(A_j,t) - alpha_j - beta_j log(B_j,t).
```

The monitor no longer divides by the simulator's known innovation parameters.
It scores the current residual against the preceding window

```text
W_t = {y_t-W, ..., y_t-1}
mean_t = sum(W_t) / W
z_t = (y_t - mean_t) / sqrt(M2_t / W),
```

then inserts (y_t). That order avoids using the observation being scored to
estimate its own center and scale. Until two nonconstant prior observations
exist, the display emits a neutral zero.

`RollingMoments` stores a fixed circular buffer, mean, and corrected sum of
squares (M_2). Adding (x) uses

```text
n'    = n + 1
delta = x - mean
mean' = mean + delta / n'
M2'   = M2 + delta (x - mean').
```

For a full window, the oldest value (x_o) is removed first:

```text
n-     = n - 1
mean-  = (n mean - x_o) / n-
M2-    = M2 - (x_o - mean)(x_o - mean-),
```

after which the ordinary addition is applied. These centered identities avoid
the cancellation-prone `sum(x*x) - sum(x)^2/n` formula. A bounded periodic
two-pass rebuild limits accumulated roundoff while keeping updates amortized
constant time. Internally, the mean is maintained as an offset from a retained
window origin, so a large common price level does not consume the precision of
small residual variation. Tests compare 20,000 rolling updates against
corrected two-pass windows at several capacities and separately exercise a
small variance around a large offset.

This remains a monitor, not a backtest or trading engine. Alpha, beta, and phi
are fixed; stationarity selection, parameter fitting, structural breaks,
costs, borrow, sizing, portfolio constraints, and execution remain outside the
example. The reusable rolling primitive can live inside an explicitly
checkpointed circuit callback when those state semantics are defined.

### Bounded top-k selection

`top_abs(values, k)` now partitions finite nonzero values in linear time and
sorts only the selected (k) entries. The prior full sort was
(O(N\log N)); the new path is expected (O(N + k\log k)). Equal magnitudes
remain deterministic by ascending slot.

## Noise-aware stat-arb ML extension

The rolling pairs monitor remains deliberately statistical and unfitted. A
separate [research harness](../experiments/statarb_ml/README.md) now tests
whether ML can add value *after* the residual, side, and candidate event have
been declared. It compares rolling-z with LightGBM and CatBoost squared-error,
Huber, quantile, and three-class triple-barrier policies.

The protocol uses five ordered blocks—fit, early-stop validation, predictive
calibration, policy selection, and untouched test—with a complete 12-step
label-horizon gap between each. Point predictions receive a calibration-only
bias correction; quantiles receive a split marginal coverage correction;
weighted multiclass outputs receive independent sigmoid probability
calibration. The selection threshold is chosen on a later block, not on the
calibrator's own observations.

Five deterministic seeds give the following controlled comparison. “Trade
t-stat” is descriptive over normalized net event returns, not annualized
Sharpe or a dependence-corrected significance result.

| Scenario and lane | Mean trade t-stat | Delta vs rolling-z | Wins | Mean total-net delta |
| --- | ---: | ---: | ---: | ---: |
| stationary OU, rolling-z | 12.666 | — | — | — |
| stationary OU, closest ML lane (LightGBM quantile) | 12.648 | -0.018 | 2/5 | -91.283 |
| observable regime stress, rolling-z | 6.626 | — | — | — |
| observable regime stress, CatBoost Huber | **7.290** | **+0.664** | **5/5** | **+48.176** |
| observable regime stress, LightGBM Huber | 7.181 | +0.555 | 5/5 | +32.926 |

The negative control prevents a boosted tree from receiving credit for merely
reconstructing a z-score. The positive control contains observable persistence
and volatility shifts plus Student-t shocks by construction; it establishes
that the harness can detect incremental conditional information. Neither
scenario is evidence of live profitability. The next required test is the
same frozen protocol over survivorship-controlled market data with
within-fold hedge estimation, bid/ask execution, borrow, and clustered or
block-bootstrap uncertainty.

## Hypercube benchmark results

The committed Criterion 0.7 harness runs with:

```bash
cargo bench -p hypercube-engine --bench performance
```

The table reports central estimates from 30 samples after a one-second warmup
and a three-second measurement target. Before/after lanes used the same
Criterion version and benchmark source. “Cells” for publisher rows are five
nodes per entity.

| Benchmark | Before / reference | Current | Result |
| --- | ---: | ---: | ---: |
| Stable five-node generation, 128 entities | 106.18 µs | 96.99 µs | 8.55% lower latency |
| Stable five-node generation, 1,024 entities | 998.31 µs | 888.30 µs | 11.59% lower latency |
| Durable five-slice publish, 128 entities | 3.308 ms | 3.314 ms | no significant change |
| Durable five-slice publish, 1,024 entities | 9.071 ms | 3.494 ms | 62.08% lower latency, 2.60× |
| Memory-mapped five-slice publish, 128 entities | — | 12.76 µs | 50.16 million cells/s |
| Memory-mapped five-slice publish, 1,024 entities | — | 109.04 µs | 46.96 million cells/s |
| Rolling z-score, allocating corrected two-pass | 156.43 µs | — | reference lane |
| Rolling z-score, reusable ring with two-pass scan | 115.52 µs | — | 1.35× versus allocating |
| Rolling z-score, online constant-time moments | — | 44.53 µs | 3.51× versus allocating; 2.59× versus ring scan |
| Top 10 of 16,384, full sort | 296.92 µs | — | reference lane |
| Top 10 of 16,384, linear select | — | 47.92 µs | 6.20× |

The memory-mapped policy is roughly 260× lower latency than durable flush at
128 entities and 32× at 1,024 on this run, but those rows have different
postconditions and must not be presented as interchangeable optimizations.

## Environment and limits

- AMD Ryzen Threadripper PRO 7985WX, 64 cores / 128 threads;
- Linux 6.8.0-136-generic, x86-64;
- Rust 1.97.1, LLVM 22.1.6;
- Cargo's standard bench profile, no `target-cpu=native` or explicit target
  features;
- Criterion 0.7, 30 samples, one-second warmup, three-second target.

The host was not isolated, frequency-locked, or core-pinned. Results are
development-host microbenchmarks, not tail-latency, contention, NUMA,
cross-process, crash-recovery, or production-SLA evidence. Re-run on the
deployment CPU with representative entity counts, readers, page residency,
flush policy, and percentile reporting before choosing wait strategies,
explicit SIMD, CPU affinity, or durability cadence.

Useful primary references are the
[Criterion analysis model](https://bheisler.github.io/criterion.rs/book/analysis.html),
the [LMAX Disruptor guide](https://lmax-exchange.github.io/disruptor/user-guide/),
the [Rust target-feature contract](https://doc.rust-lang.org/stable/reference/attributes/codegen.html),
the current [nightly-only portable SIMD API](https://doc.rust-lang.org/nightly/std/simd/struct.Simd.html),
the [`memmap2` flush contract](https://docs.rs/memmap2/latest/memmap2/struct.MmapMut.html),
Welford's [corrected-sums update](https://doi.org/10.1080/00401706.1962.10490022),
and Chan, Golub, and LeVeque's
[variance analysis](https://doi.org/10.1080/00031305.1983.10483115).
