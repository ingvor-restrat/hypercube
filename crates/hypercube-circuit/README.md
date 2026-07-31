# Hypercube Circuit

`hypercube-circuit` adds stateful generation processors and deterministic
record/replay around the pure `hypercube-engine`.

The initial processor detects persistent threshold transitions over calculated
Hypercube nodes. Recordings contain complete accepted engine updates, semantic
snapshot digests, complete trigger-state cross-sections, and transition events.
The dense state can be projected into Slice vectors while the sparse
transitions feed subscribers. A fresh engine and fresh circuit can replay the
recording without invoking live effects.

The recording API is transport-neutral. The included JSON Lines adapter is
useful for tests, examples, and inspection; an Aeron Archive adapter can map
one manifest and one generation record to application messages without
changing replay semantics.

See the repository
[replay guide](https://github.com/ingvor-restrat/hypercube/blob/main/docs/replay.md)
for the complete contract and runnable example.
