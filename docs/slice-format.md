# Slice format

A slice is a typed memory-mapped vector aligned to a versioned entity layout.
Version 1 uses a 256-byte header followed by a fixed-size payload.

## Header

| Field | Purpose |
| --- | --- |
| magic and version | Reject unknown physical formats. |
| value type and slot size | Define the payload representation. |
| capacity and active length | Define the vector bounds. |
| layout and schema hashes | Reject incompatible vector operations. |
| writer epoch | Detect an in-progress whole-vector update. |
| heartbeat timestamp | Show that the owning writer is alive. |
| created and updated timestamps | Separate file age from data age. |

All current payloads require a little-endian target. Dense `f64` slices use a
strip-level even/odd epoch around vector updates. Fixed quote, trade, and
quote-at-trade records use an even/odd sequence counter for each slot.

## Layout

A JSON layout maps a stable entity name and instrument identity to a dense
slot:

```json
{
  "layout_id": "example-v1",
  "asset_class": "example",
  "capacity": 2,
  "symbols": [
    {
      "slot_id": 0,
      "symbol": "A",
      "instrument_id": "example:A",
      "enabled": true
    }
  ]
}
```

Slots do not move inside a layout. Reordering or compaction creates a new
layout and therefore a new layout hash.

## Reader guarantees

A successful point read observed one stable fixed record. A successful vector
snapshot or dot product observed one stable epoch of each participating slice.
Separate input slices may still come from different logical generations.

The implementation retries optimistic reads a bounded number of times and
returns an error rather than accepting a value that may have torn.

