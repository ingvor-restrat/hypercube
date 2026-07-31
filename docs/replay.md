# Deterministic Record and Replay

`hypercube-circuit` surrounds the pure Hypercube engine with ordered, stateful
generation processing and a versioned replay contract.

The companion
[Part II paper](latex/hypercube_circuits.pdf) develops the architecture,
failure contracts, applications, and executable evidence in one narrative.

```text
live inputs
    |
    v
generation assembler ---- complete Update(g) ----> recording / Aeron Archive
                              |
                              v
                       Hypercube engine
                              |
                              v
                       coherent Snapshot(g)
                              |
                              v
                   single-producer Disruptor
                              |
                              v
                     stateful processors
                              |
                              v
                 state, transitions, digest

recording ----> fresh engine + fresh processor state ----> exactness report
```

## Implemented replay boundary

The current boundary is one complete, serialized `Update` per accepted
generation. A recording starts with a manifest and continues with ordered
generation records:

- the manifest identifies the run, build, configuration, entity layout, and
  stateful trigger definitions;
- each generation carries the complete `Update`, its circuit sequence,
  optional upstream source positions, the semantic snapshot digest, and the
  expected state cross-section and transitions.

The first storage adapter is streaming JSON Lines. It is intended for
development, audit, and schema evolution rather than the final low-latency
wire encoding. JSON parsing enables `serde_json`'s `float_roundtrip` feature,
so finite IEEE-754 input values reproduce their exact bit patterns. The
adapter rejects NaN and infinity before calculation; represent missing
primitive values by omitting their fields.

## What exact replay means

The snapshot digest includes:

- generation and logical observation time;
- entity identity and ordering;
- cell identity, status, and exact floating-point bits.

It deliberately excludes:

- `ExecutionMode` (`Live` becomes `Replay`);
- measured compute duration;
- wall-clock time, thread identity, and transport timing.

Replay constructs a fresh engine and fresh callback state, processes every
generation in lockstep, and compares circuit sequence, semantic digest, and
ordered state and transitions. A mismatch is reported at the first divergent
generation while the rest of the recording is still checked.

## Stateful processing

The initial processor is a persistent upper-threshold trigger. It supports:

- entry only after a configurable number of consecutive qualifying
  generations;
- hysteresis, so an active signal exits at a lower threshold;
- explicit invalidation when its source value becomes missing;
- logical time from `observed_at_ms`, never processor wall time.

Every frame contains both a complete ordered trigger-state cross-section and
the sparse transitions for that generation. The former maps naturally to a
new Slice vector and lets late subscribers bootstrap; the latter is the
efficient event stream for already synchronized clients.

`DisruptorCircuit` acknowledges each generation after all configured
processing completes. This lockstep policy is the deterministic baseline.
Independent lossy consumers can be added outside that correctness path for
dashboards or telemetry. The default wait strategy sleeps briefly to keep
development CPU use modest; production deployments can select spin-loop or
busy-spin behavior explicitly when the latency budget justifies a dedicated
core.

## Run the example

Record 40 synthetic generations across 32 entities:

```bash
cargo run -p hypercube-circuit --bin hypercube-replay -- \
  record-demo /tmp/hypercube-factor.jsonl 40 32
```

Replay the entire calculation and trigger state:

```bash
cargo run -p hypercube-circuit --bin hypercube-replay -- \
  verify /tmp/hypercube-factor.jsonl
```

`verify` exits with status 0 for an exact replay, 2 for semantic divergence,
and 1 for an operational or malformed-recording error.

## Aeron and Archive mapping

The recording structs are transport-neutral envelopes. An Aeron adapter can
encode the same two logical message types with SBE or another stable binary
schema:

```text
hypercube.recording.manifest.v1
hypercube.recording.generation.v1
```

Store the Archive recording identity and start position in manifest metadata.
Store each input stream's last included Archive position in
`source_positions`. Those frontiers make a generation a reproducible cut
across multiple feeds.

The boundary determines what can be reconstructed:

- archive factor frames to replay only downstream trigger and strategy logic;
- archive complete Hypercube updates to recalculate factors and callbacks;
- archive normalized raw events plus generation seals to reproduce ingestion,
  generation assembly, factors, and callbacks.

The current implementation chooses complete updates: it is substantially more
useful than factor-only replay without yet defining every raw-feed event and
watermark rule.

## Replay isolation

Replay output should use a separate namespace such as
`replay/{run_id}/...`. An execution adapter must reject replay-mode data. The
MVP contains no order or other external-effect authority, so verification
cannot accidentally trade.

## Natural extensions

- an Aeron Archive/SBE recording adapter;
- canonical sequencing and generation seals for multiple raw input streams;
- state checkpoints for seeking into long recordings;
- immutable deployment manifests containing binaries and full configuration;
- counterfactual replay that intentionally changes a graph or trigger and
  compares outcomes;
- idempotent, separately audited adapters for authorized external effects.
