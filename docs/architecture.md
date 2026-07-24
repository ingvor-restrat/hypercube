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
          |         |
          v         v
     .slice files   API/SSE views
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

## Publication model

A snapshot is coherent in process. `SlicePublisher` projects selected node
cross-sections into independently stabilized memory-mapped vectors. Readers
therefore get a coherent snapshot of one slice, not an atomic transaction
across all slices.

Applications requiring multi-slice decision consistency should consume the
in-process snapshot or add a generation manifest and reader-pinning protocol.
That stronger protocol is intentionally not implied by version 1.

## Extraction boundary

The public engine contains reusable mechanics:

- stable layouts and catalogs;
- fixed-record and dense-vector memory mappings;
- single-writer/many-reader consistency;
- field extraction and cross-sectional transforms;
- topological linear composition;
- generation monotonicity and deterministic snapshots;
- synthetic injection and visualization.

Vendor feeds, transports, persistence adapters, proprietary factor
definitions, live host operations, portfolio construction, and execution
authority stay outside this repository.

