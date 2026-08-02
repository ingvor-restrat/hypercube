# Architecture

Hypercube separates current-state publication from deterministic calculation.
The distinction keeps common reads cheap without pretending that memory
mapping is a database, message bus, or globally atomic transaction.

```text
synthetic or external input
           |
           v
   coherent generation
           |
           v
  Hypercube dependency graph
     |       |        |
     v       v        v
  field   composite  normalized
   node      node       node
     \        |        /
      +-------+-------+
              |
              v
       immutable snapshot
       |       |          |
       v       v          v
  .slice files API/SSE  Disruptor circuit
                           |
                           v
                    state/transitions
```

## Coordinate model

The current engine makes four dimensions explicit:

- **entity** — a stable row key;
- **field** — a primitive value in an input row;
- **node** — a computed cross-section;
- **generation** — the coherent input and output boundary.

Transforms such as rank, percentile, z-score, and rank-z-score operate across
the entity dimension. Linear nodes combine already resolved node
cross-sections. Required dependencies must be available for an entity;
optional dependencies may be absent. When weight normalization is enabled, a
linear node normalizes by the absolute weights actually available to that
entity.

The graph is pure: it emits calculated snapshots but has no authority to
perform external actions.

Node configuration is compiled separately from value evaluation. An unchanged
declaration reuses indexed dependency edges and topological order while every
generation still recalculates its values. This removes repeated planning
without turning the engine into a cross-generation value cache.

`hypercube-circuit` preserves that boundary. It receives an already coherent
snapshot and provides ordered, stateful processing with logical generation
time. Replay creates fresh engine and processor state. Transports, persistence,
and any external-effect adapters remain outside the graph.

## Publication model

A snapshot is coherent in process. `SlicePublisher` projects selected node
cross-sections into independently stabilized memory-mapped vectors. Readers
therefore get a coherent snapshot of one slice, not an atomic transaction
across all slices.

Publisher projection is one pass into preallocated node vectors with cached
entity slots. Mapped-memory visibility, asynchronous flush, and synchronous
durability are separate policies. The live synthetic server selects
visibility; the compatibility `publish()` call selects durability. None
changes the independent-slice atomicity boundary.

Applications requiring multi-slice decision consistency should consume the
in-process snapshot or add a generation manifest and reader-pinning protocol.
That stronger protocol is intentionally not implied by version 1.

## Replay model

The implemented replay boundary records each complete Hypercube `Update`, the
resulting semantic snapshot digest, and complete trigger state plus
transitions. The digest ignores live-versus-replay mode and runtime timing
while retaining exact value bits, statuses, entity ordering, generation, and
observation time.

The development adapter stores versioned JSON Lines. The same manifest and
generation envelopes map to Aeron Archive messages; source positions identify
the last event from each upstream stream included in a generation. Replay
outputs belong in a separate namespace and must not reach execution authority.
See the [record/replay contract](replay.md).

## Backpressure and truth

The ring is an execution boundary, not the system of record, and does not make
an overloaded consumer correct by itself. Nonblocking circuit submission
returns `RingFull`; it never silently overwrites an unprocessed generation.
The owner must then apply an explicit policy:

- slow or backpressure the producer on a correctness-critical lane;
- retain the raw input in Aeron Archive and catch up from its position;
- coalesce or drop only a separately identified best-effort view.

`CaptureSession` uses lockstep processing and recording, so its engine does not
advance again until the stateful generation completes. Where the market-data
source cannot be slowed, archive the normalized input first and let generation
assembly advance from that durable log.

`RollingMoments` provides fixed-capacity online mean, variance, and z-score
state for an explicit owner. It does not silently add history to the pure
graph. A callback that persists this state must still define its checkpoint and
replay schema.

## Extraction boundary

The public engine contains reusable mechanics:

- stable layouts and catalogs;
- fixed-record and dense-vector memory mappings;
- single-writer/many-reader consistency;
- field extraction and cross-sectional transforms;
- topological linear composition;
- generation monotonicity and deterministic snapshots;
- reusable graph plans and fixed-capacity rolling moments;
- synthetic injection and visualization.

Vendor feeds, transports, persistence adapters, proprietary factor
definitions, live host operations, portfolio construction, and execution
authority stay outside this repository.
