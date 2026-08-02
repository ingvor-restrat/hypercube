//! Typed memory-mapped vectors for the current working set of a live system.
//!
//! A layout gives stable entities dense slots. A slice publishes one fixed
//! value schema over that layout. Version 1 supports coherent `f64` vector
//! snapshots and individually coherent fixed-record point reads with one
//! writer and many readers.

#![warn(missing_docs)]
#![cfg_attr(target_endian = "big", allow(dead_code))]

#[cfg(target_endian = "big")]
compile_error!(
    "hypercube_slice v1 stores native f64 payloads and currently requires little-endian targets"
);

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::mem;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use memmap2::{Mmap, MmapMut, MmapOptions};
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh64::xxh64;

const MAGIC: &[u8; 8] = b"HCUBSLCE";
const VERSION: u32 = 1;
/// Byte length reserved for the version-1 file header.
pub const HEADER_SIZE: usize = 256;

const OFF_MAGIC: usize = 0;
const OFF_VERSION: usize = 8;
const OFF_HEADER_LEN: usize = 12;
const OFF_VALUE_TYPE: usize = 16;
const OFF_FLAGS: usize = 20;
const OFF_CAPACITY: usize = 24;
const OFF_ACTIVE_LEN: usize = 32;
const OFF_SLOT_SIZE: usize = 40;
const OFF_LAYOUT_HASH: usize = 48;
const OFF_SCHEMA_HASH: usize = 56;
const OFF_WRITER_EPOCH: usize = 64;
const OFF_HEARTBEAT_NS: usize = 72;
const OFF_CREATED_NS: usize = 80;
const OFF_UPDATED_NS: usize = 88;

const FLAG_WRITABLE_OWNER: u32 = 1;

/// Physical payload schema stored in a slice file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueType {
    /// Dense IEEE-754 double-precision vector.
    F64 = 1,
    /// Per-slot [`QuoteV1`] wrapped in a sequence counter.
    QuoteV1 = 100,
    /// Per-slot [`TradeV1`] wrapped in a sequence counter.
    TradeV1 = 101,
    /// Per-slot [`TaqV1`] wrapped in a sequence counter.
    TaqV1 = 102,
}

impl ValueType {
    /// Decode the stable integer discriminator stored in a file header.
    pub fn from_u32(value: u32) -> Result<Self> {
        match value {
            1 => Ok(Self::F64),
            100 => Ok(Self::QuoteV1),
            101 => Ok(Self::TradeV1),
            102 => Ok(Self::TaqV1),
            _ => Err(anyhow!("unsupported slice value type: {value}")),
        }
    }

    /// Return the byte width of one physical slot.
    pub fn slot_size(self) -> u64 {
        match self {
            Self::F64 => mem::size_of::<f64>() as u64,
            Self::QuoteV1 => mem::size_of::<SequencedRecord<QuoteV1>>() as u64,
            Self::TradeV1 => mem::size_of::<SequencedRecord<TradeV1>>() as u64,
            Self::TaqV1 => mem::size_of::<SequencedRecord<TaqV1>>() as u64,
        }
    }

    /// Return the stable versioned hash for this payload schema.
    pub fn schema_hash(self) -> u64 {
        match self {
            Self::F64 => xxh64(b"hypercube.slice.value.f64.v1", 0),
            Self::QuoteV1 => xxh64(b"hypercube.slice.value.quote.v1", 0),
            Self::TradeV1 => xxh64(b"hypercube.slice.value.trade.v1", 0),
            Self::TaqV1 => xxh64(b"hypercube.slice.value.taq.v1", 0),
        }
    }
}

/// The fixed record contains a usable observation.
pub const RECORD_FLAG_VALID: u64 = 1 << 0;
/// The trade record was classified as buyer initiated.
pub const RECORD_FLAG_TRADE_BUY: u64 = 1 << 1;
/// The trade record was classified as seller initiated.
pub const RECORD_FLAG_TRADE_SELL: u64 = 1 << 2;
/// A TAQ record includes prevailing quote context.
pub const TAQ_FLAG_HAS_QUOTE: u64 = 1 << 8;
/// The attached quote exceeded the configured maximum age.
pub const TAQ_FLAG_QUOTE_STALE: u64 = 1 << 9;
/// The attached quote timestamp is later than the trade timestamp.
pub const TAQ_FLAG_QUOTE_AFTER_TRADE: u64 = 1 << 10;

/// Version-1 top-of-book quote payload.
///
/// The fixed, pointer-free `repr(C)` layout is suitable for a shared slice
/// slot. Prices and quantities retain the producer's declared units.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct QuoteV1 {
    /// Source or exchange event time in Unix-epoch nanoseconds.
    pub exchange_ts_ns: i64,
    /// Local ingestion time in Unix-epoch nanoseconds.
    pub ingest_ts_ns: i64,
    /// Best bid price.
    pub bid_px: f64,
    /// Quantity available at the best bid.
    pub bid_qty: f64,
    /// Best ask price.
    pub ask_px: f64,
    /// Quantity available at the best ask.
    pub ask_qty: f64,
    /// Arithmetic midpoint when both sides are positive and finite.
    pub mid_px: f64,
    /// Bid/ask spread in basis points of midpoint.
    pub spread_bps: f64,
    /// Record-quality bits such as [`RECORD_FLAG_VALID`].
    pub flags: u64,
}

impl QuoteV1 {
    /// Construct a valid quote and derive midpoint and spread.
    pub fn new(
        exchange_ts_ns: i64,
        ingest_ts_ns: i64,
        bid_px: f64,
        bid_qty: f64,
        ask_px: f64,
        ask_qty: f64,
    ) -> Self {
        let mid_px = if bid_px.is_finite() && ask_px.is_finite() && bid_px > 0.0 && ask_px > 0.0 {
            0.5 * (bid_px + ask_px)
        } else {
            f64::NAN
        };
        let spread_bps = if mid_px.is_finite() && mid_px > 0.0 {
            (ask_px - bid_px) / mid_px * 10_000.0
        } else {
            f64::NAN
        };
        Self {
            exchange_ts_ns,
            ingest_ts_ns,
            bid_px,
            bid_qty,
            ask_px,
            ask_qty,
            mid_px,
            spread_bps,
            flags: RECORD_FLAG_VALID,
        }
    }

    /// Return whether [`RECORD_FLAG_VALID`] is set.
    pub fn is_valid(&self) -> bool {
        self.flags & RECORD_FLAG_VALID != 0
    }
}

/// Version-1 trade payload.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TradeV1 {
    /// Source or exchange event time in Unix-epoch nanoseconds.
    pub exchange_ts_ns: i64,
    /// Upstream system timestamp in Unix-epoch nanoseconds.
    pub system_ts_ns: i64,
    /// Local ingestion time in Unix-epoch nanoseconds.
    pub ingest_ts_ns: i64,
    /// Execution price.
    pub px: f64,
    /// Unsigned execution quantity.
    pub qty: f64,
    /// Positive buy, negative sell, or zero unclassified quantity.
    pub signed_qty: f64,
    /// Validity and trade-side classification bits.
    pub flags: u64,
}

impl TradeV1 {
    /// Construct a valid trade and classify a recognized textual side.
    pub fn new(
        exchange_ts_ns: i64,
        system_ts_ns: i64,
        ingest_ts_ns: i64,
        px: f64,
        qty: f64,
        side: Option<&str>,
    ) -> Self {
        let mut flags = RECORD_FLAG_VALID;
        let mut signed_qty = 0.0;
        if let Some(side) = side {
            match side.trim().to_ascii_lowercase().as_str() {
                "buy" | "b" | "bid" | "ask_lift" | "lift" => {
                    flags |= RECORD_FLAG_TRADE_BUY;
                    signed_qty = qty.abs();
                }
                "sell" | "s" | "ask" | "bid_hit" | "hit" => {
                    flags |= RECORD_FLAG_TRADE_SELL;
                    signed_qty = -qty.abs();
                }
                _ => {}
            }
        }
        Self {
            exchange_ts_ns,
            system_ts_ns,
            ingest_ts_ns,
            px,
            qty,
            signed_qty,
            flags,
        }
    }

    /// Return whether [`RECORD_FLAG_VALID`] is set.
    pub fn is_valid(&self) -> bool {
        self.flags & RECORD_FLAG_VALID != 0
    }
}

/// Version-1 trade-and-prevailing-quote payload.
///
/// TAQ retains both source timestamps and quality flags so a consumer can
/// decide whether the local quote-at-trade join is suitable for its purpose.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TaqV1 {
    /// Trade source timestamp in Unix-epoch nanoseconds.
    pub trade_exchange_ts_ns: i64,
    /// Trade upstream-system timestamp in Unix-epoch nanoseconds.
    pub trade_system_ts_ns: i64,
    /// Trade local-ingestion timestamp in Unix-epoch nanoseconds.
    pub trade_ingest_ts_ns: i64,
    /// Quote source timestamp in Unix-epoch nanoseconds.
    pub quote_exchange_ts_ns: i64,
    /// Quote local-ingestion timestamp in Unix-epoch nanoseconds.
    pub quote_ingest_ts_ns: i64,
    /// Trade source time minus quote source time.
    pub quote_age_ns: i64,
    /// Execution price.
    pub trade_px: f64,
    /// Unsigned execution quantity.
    pub trade_qty: f64,
    /// Signed execution quantity copied from the trade.
    pub signed_qty: f64,
    /// Prevailing best bid price.
    pub bid_px: f64,
    /// Prevailing best bid quantity.
    pub bid_qty: f64,
    /// Prevailing best ask price.
    pub ask_px: f64,
    /// Prevailing best ask quantity.
    pub ask_qty: f64,
    /// Prevailing quote midpoint.
    pub mid_px: f64,
    /// Prevailing quote spread in basis points.
    pub spread_bps: f64,
    /// Validity, trade-side, and quote-quality bits.
    pub flags: u64,
}

impl TaqV1 {
    /// Join a trade to an optional prevailing quote and set age-quality flags.
    pub fn from_trade_and_quote(
        trade: TradeV1,
        quote: Option<QuoteV1>,
        max_quote_age_ns: i64,
    ) -> Self {
        let mut out = Self {
            trade_exchange_ts_ns: trade.exchange_ts_ns,
            trade_system_ts_ns: trade.system_ts_ns,
            trade_ingest_ts_ns: trade.ingest_ts_ns,
            trade_px: trade.px,
            trade_qty: trade.qty,
            signed_qty: trade.signed_qty,
            flags: RECORD_FLAG_VALID
                | (trade.flags & (RECORD_FLAG_TRADE_BUY | RECORD_FLAG_TRADE_SELL)),
            ..Self::default()
        };
        if let Some(quote) = quote.filter(QuoteV1::is_valid) {
            out.quote_exchange_ts_ns = quote.exchange_ts_ns;
            out.quote_ingest_ts_ns = quote.ingest_ts_ns;
            out.quote_age_ns = trade.exchange_ts_ns.saturating_sub(quote.exchange_ts_ns);
            out.bid_px = quote.bid_px;
            out.bid_qty = quote.bid_qty;
            out.ask_px = quote.ask_px;
            out.ask_qty = quote.ask_qty;
            out.mid_px = quote.mid_px;
            out.spread_bps = quote.spread_bps;
            out.flags |= TAQ_FLAG_HAS_QUOTE;
            if out.quote_age_ns < 0 {
                out.flags |= TAQ_FLAG_QUOTE_AFTER_TRADE;
            }
            if max_quote_age_ns > 0 && out.quote_age_ns > max_quote_age_ns {
                out.flags |= TAQ_FLAG_QUOTE_STALE;
            }
        } else {
            out.quote_age_ns = i64::MAX;
            out.mid_px = f64::NAN;
            out.spread_bps = f64::NAN;
        }
        out
    }

    /// Return whether the record contains a quote.
    pub fn has_quote(&self) -> bool {
        self.flags & TAQ_FLAG_HAS_QUOTE != 0
    }

    /// Return whether the quote exceeded the configured maximum age.
    pub fn quote_is_stale(&self) -> bool {
        self.flags & TAQ_FLAG_QUOTE_STALE != 0
    }

    /// Return whether the quote source timestamp follows the trade.
    pub fn quote_is_after_trade(&self) -> bool {
        self.flags & TAQ_FLAG_QUOTE_AFTER_TRADE != 0
    }
}

/// Marker for fixed-size, pointer-free records supported by struct slices.
pub trait SliceRecord: Copy + Default + 'static {
    /// Physical schema discriminator for this record.
    const VALUE_TYPE: ValueType;
}

impl SliceRecord for QuoteV1 {
    const VALUE_TYPE: ValueType = ValueType::QuoteV1;
}

impl SliceRecord for TradeV1 {
    const VALUE_TYPE: ValueType = ValueType::TradeV1;
}

impl SliceRecord for TaqV1 {
    const VALUE_TYPE: ValueType = ValueType::TaqV1;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SequencedRecord<T: SliceRecord> {
    seq: u64,
    value: T,
}

/// One stable entity-to-slot assignment in a layout.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LayoutSymbol {
    /// Dense zero-based payload index.
    pub slot_id: u32,
    /// Normalized display or lookup key.
    pub symbol: String,
    /// Stable namespaced entity or instrument identifier.
    pub instrument_id: String,
    /// Whether lookups should currently resolve this row.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_enabled() -> bool {
    true
}

/// Versioned mapping from stable entity keys to dense slice slots.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LayoutRegistry {
    /// Caller-defined identity for this immutable slot assignment.
    pub layout_id: String,
    /// Normalized namespace; retained as `asset_class` in the version-1 JSON.
    pub asset_class: String,
    /// Allocated payload slots, including unused or disabled slots.
    pub capacity: u32,
    /// Declared slot assignments.
    pub symbols: Vec<LayoutSymbol>,
}

impl LayoutRegistry {
    /// Construct an unchecked layout value; call [`Self::validate`] before use.
    pub fn new(
        layout_id: impl Into<String>,
        asset_class: impl Into<String>,
        capacity: u32,
        symbols: Vec<LayoutSymbol>,
    ) -> Self {
        Self {
            layout_id: layout_id.into(),
            asset_class: asset_class.into(),
            capacity,
            symbols,
        }
    }

    /// Build and validate a layout from financial-symbol-style keys.
    pub fn from_symbols(
        layout_id: impl Into<String>,
        asset_class: impl Into<String>,
        capacity: u32,
        symbols: &[String],
    ) -> Result<Self> {
        Self::from_symbols_with_instrument_prefix(layout_id, asset_class, capacity, symbols, None)
    }

    /// Domain-neutral spelling of [`Self::from_symbols`].
    pub fn from_entities(
        layout_id: impl Into<String>,
        namespace: impl Into<String>,
        capacity: u32,
        entity_keys: &[String],
    ) -> Result<Self> {
        Self::from_symbols(layout_id, namespace, capacity, entity_keys)
    }

    /// Build a symbol layout with a caller-supplied identifier prefix.
    pub fn from_symbols_with_instrument_prefix(
        layout_id: impl Into<String>,
        asset_class: impl Into<String>,
        capacity: u32,
        symbols: &[String],
        instrument_prefix: Option<&str>,
    ) -> Result<Self> {
        if symbols.len() > capacity as usize {
            bail!(
                "symbol count {} exceeds layout capacity {}",
                symbols.len(),
                capacity
            );
        }
        let asset_class = normalize_asset_class(&asset_class.into())?;
        let mut rows = Vec::with_capacity(symbols.len());
        for (slot_id, raw) in symbols.iter().enumerate() {
            let symbol = normalize_symbol(raw)?;
            rows.push(LayoutSymbol {
                slot_id: slot_id as u32,
                instrument_id: default_instrument_id(&asset_class, &symbol, instrument_prefix)?,
                symbol: symbol.clone(),
                enabled: true,
            });
        }
        let layout = Self::new(layout_id, asset_class, capacity, rows);
        layout.validate()?;
        Ok(layout)
    }

    /// Load, parse, and validate a layout JSON document.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).with_context(|| format!("failed reading {}", path.display()))?;
        let layout: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed parsing layout {}", path.display()))?;
        layout.validate()?;
        Ok(layout)
    }

    /// Validate and save a human-readable layout JSON document.
    pub fn save_pretty(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let path = path.as_ref();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed creating {}", parent.display()))?;
        }
        let body = serde_json::to_vec_pretty(self).context("failed serializing layout")?;
        fs::write(path, body).with_context(|| format!("failed writing {}", path.display()))?;
        Ok(())
    }

    /// Check normalization, capacity, uniqueness, and slot bounds.
    pub fn validate(&self) -> Result<()> {
        if self.layout_id.trim().is_empty() {
            bail!("layout_id cannot be empty");
        }
        if self.asset_class.trim().is_empty() {
            bail!("asset_class cannot be empty");
        }
        let normalized_asset_class = normalize_asset_class(&self.asset_class)?;
        if normalized_asset_class != self.asset_class {
            bail!(
                "asset_class '{}' should be normalized as '{}'",
                self.asset_class,
                normalized_asset_class
            );
        }
        let mut slots = HashSet::with_capacity(self.symbols.len());
        let mut symbols = HashSet::with_capacity(self.symbols.len());
        let mut instruments = HashSet::with_capacity(self.symbols.len());
        for row in &self.symbols {
            if row.slot_id >= self.capacity {
                bail!(
                    "slot_id {} for {} exceeds capacity {}",
                    row.slot_id,
                    row.symbol,
                    self.capacity
                );
            }
            if !slots.insert(row.slot_id) {
                bail!("duplicate slot_id {}", row.slot_id);
            }
            let symbol = normalize_symbol(&row.symbol)?;
            if symbol != row.symbol {
                bail!(
                    "symbol '{}' should be normalized as '{}'",
                    row.symbol,
                    symbol
                );
            }
            if !symbols.insert(symbol.clone()) {
                bail!("duplicate symbol {symbol}");
            }
            if row.instrument_id.trim().is_empty() {
                bail!("instrument_id cannot be empty for {symbol}");
            }
            if !instruments.insert(row.instrument_id.clone()) {
                bail!("duplicate instrument_id {}", row.instrument_id);
            }
        }
        Ok(())
    }

    /// Hash the validated canonical JSON layout.
    pub fn layout_hash(&self) -> Result<u64> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).context("failed serializing layout for hash")?;
        Ok(xxh64(&bytes, 0))
    }

    /// Return one plus the highest enabled slot, or zero for an empty layout.
    pub fn active_len(&self) -> u64 {
        self.symbols
            .iter()
            .filter(|row| row.enabled)
            .map(|row| row.slot_id as u64 + 1)
            .max()
            .unwrap_or(0)
    }

    /// Resolve a dense slot to its declared symbol, including disabled rows.
    pub fn symbol_for_slot(&self, slot_id: u32) -> Option<&str> {
        self.symbols
            .iter()
            .find(|row| row.slot_id == slot_id)
            .map(|row| row.symbol.as_str())
    }

    /// Domain-neutral spelling of [`Self::symbol_for_slot`].
    pub fn entity_for_slot(&self, slot_id: u32) -> Option<&str> {
        self.symbol_for_slot(slot_id)
    }

    /// Resolve a normalized enabled symbol to a dense slot.
    pub fn slot_for_symbol(&self, symbol: &str) -> Result<Option<u32>> {
        let normalized = normalize_symbol(symbol)?;
        Ok(self
            .symbols
            .iter()
            .find(|row| row.symbol == normalized && row.enabled)
            .map(|row| row.slot_id))
    }

    /// Domain-neutral spelling of [`Self::slot_for_symbol`].
    pub fn slot_for_entity(&self, entity_key: &str) -> Result<Option<u32>> {
        self.slot_for_symbol(entity_key)
    }

    /// Return enabled normalized symbols keyed to their dense slots.
    pub fn slot_map(&self) -> HashMap<String, u32> {
        self.symbols
            .iter()
            .filter(|row| row.enabled)
            .map(|row| (row.symbol.clone(), row.slot_id))
            .collect()
    }
}

/// Domain-neutral name for a stable slice layout.
pub type EntityLayout = LayoutRegistry;

/// Domain-neutral name for one row in a stable slice layout.
pub type LayoutEntity = LayoutSymbol;

/// Discoverable metadata for one named slice file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SliceCatalogEntry {
    /// Stable logical name used by consumers.
    pub name: String,
    /// Normalized namespace shared with the layout.
    pub asset_class: String,
    /// Identity of the associated layout.
    pub layout_id: String,
    /// Hash of the exact associated layout.
    pub layout_hash: u64,
    /// Physical payload schema.
    pub value_type: ValueType,
    /// File path as published by the writer.
    pub path: String,
    /// Caller-defined purpose such as `node_output` or `latest`.
    pub role: String,
    /// Optional human-readable meaning and units.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl SliceCatalogEntry {
    /// Check name, namespace, layout identity, path, and role.
    pub fn validate(&self) -> Result<()> {
        validate_slice_name(&self.name)?;
        let asset_class = normalize_asset_class(&self.asset_class)?;
        if asset_class != self.asset_class {
            bail!(
                "catalog entry asset_class '{}' should be normalized as '{}'",
                self.asset_class,
                asset_class
            );
        }
        if self.layout_id.trim().is_empty() {
            bail!("catalog entry {} has empty layout_id", self.name);
        }
        if self.layout_hash == 0 {
            bail!("catalog entry {} has zero layout_hash", self.name);
        }
        if self.path.trim().is_empty() {
            bail!("catalog entry {} has empty path", self.name);
        }
        if self.role.trim().is_empty() {
            bail!("catalog entry {} has empty role", self.name);
        }
        Ok(())
    }
}

/// Versioned collection of discoverable slice files.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SliceCatalog {
    /// Catalog schema version; currently `1`.
    pub version: u32,
    /// Entries ordered by logical name when modified through [`Self::upsert`].
    pub entries: Vec<SliceCatalogEntry>,
}

impl Default for SliceCatalog {
    fn default() -> Self {
        Self {
            version: 1,
            entries: Vec::new(),
        }
    }
}

impl SliceCatalog {
    /// Load and validate a catalog, returning an empty catalog if absent.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::default());
        }
        let bytes = fs::read(path).with_context(|| format!("failed reading {}", path.display()))?;
        let catalog: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed parsing catalog {}", path.display()))?;
        catalog.validate()?;
        Ok(catalog)
    }

    /// Validate and save a human-readable catalog JSON document.
    pub fn save_pretty(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let path = path.as_ref();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed creating {}", parent.display()))?;
        }
        let body = serde_json::to_vec_pretty(self).context("failed serializing catalog")?;
        fs::write(path, body).with_context(|| format!("failed writing {}", path.display()))?;
        Ok(())
    }

    /// Check schema version, entry validity, and unique names.
    pub fn validate(&self) -> Result<()> {
        if self.version != 1 {
            bail!("unsupported catalog version {}", self.version);
        }
        let mut names = HashSet::with_capacity(self.entries.len());
        for entry in &self.entries {
            entry.validate()?;
            if !names.insert(entry.name.clone()) {
                bail!("duplicate catalog entry {}", entry.name);
            }
        }
        Ok(())
    }

    /// Insert or replace an entry and restore lexical name order.
    pub fn upsert(&mut self, entry: SliceCatalogEntry) -> Result<()> {
        entry.validate()?;
        if let Some(existing) = self.entries.iter_mut().find(|row| row.name == entry.name) {
            *existing = entry;
        } else {
            self.entries.push(entry);
            self.entries.sort_by(|a, b| a.name.cmp(&b.name));
        }
        Ok(())
    }

    /// Find one entry by its exact logical name.
    pub fn find(&self, name: &str) -> Option<&SliceCatalogEntry> {
        self.entries.iter().find(|entry| entry.name == name)
    }

    /// Select every entry in a normalized namespace.
    pub fn by_asset_class<'a>(&'a self, asset_class: &str) -> Result<Vec<&'a SliceCatalogEntry>> {
        let asset_class = normalize_asset_class(asset_class)?;
        Ok(self
            .entries
            .iter()
            .filter(|entry| entry.asset_class == asset_class)
            .collect())
    }
}

/// Decoded metadata from a version-1 slice file header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceHeaderInfo {
    /// Physical format version.
    pub version: u32,
    /// Header byte length.
    pub header_len: u32,
    /// Payload schema.
    pub value_type: ValueType,
    /// File-level capability flags.
    pub flags: u32,
    /// Total allocated slots.
    pub capacity: u64,
    /// Logical payload prefix used by vector operations.
    pub active_len: u64,
    /// Physical bytes occupied by one slot.
    pub slot_size: u64,
    /// Hash of the associated entity layout.
    pub layout_hash: u64,
    /// Hash of the versioned payload schema.
    pub schema_hash: u64,
    /// Even/odd whole-vector publication epoch.
    pub writer_epoch: u64,
    /// Last writer liveness update in Unix-epoch nanoseconds.
    pub heartbeat_ns: u64,
    /// File creation time in Unix-epoch nanoseconds.
    pub created_ns: u64,
    /// Last payload update in Unix-epoch nanoseconds.
    pub updated_ns: u64,
}

impl SliceHeaderInfo {
    /// Return the checked payload byte length for this platform.
    pub fn payload_len(&self) -> Result<usize> {
        let bytes = self
            .capacity
            .checked_mul(self.slot_size)
            .ok_or_else(|| anyhow!("slice payload size overflow"))?;
        usize::try_from(bytes).context("slice payload is too large for this platform")
    }

    /// Return the checked total file length.
    pub fn file_len(&self) -> Result<u64> {
        self.capacity
            .checked_mul(self.slot_size)
            .and_then(|payload| payload.checked_add(HEADER_SIZE as u64))
            .ok_or_else(|| anyhow!("slice file size overflow"))
    }
}

/// Read-only mapping of a dense `f64` slice.
///
/// Guarded operations retry until they observe one unchanged even writer
/// epoch or return an error after the bounded retry budget.
pub struct F64SliceReader {
    path: PathBuf,
    mmap: Mmap,
    header: SliceHeaderInfo,
}

impl F64SliceReader {
    /// Open and validate an existing `f64` slice.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file =
            File::open(&path).with_context(|| format!("failed opening {}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("failed mmap read-only {}", path.display()))?;
        let header = read_header(&mmap)?;
        validate_f64_header(&header)?;
        validate_file_len(mmap.len(), &header)?;
        Ok(Self { path, mmap, header })
    }

    /// Return the mapped file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read header metadata, refreshing the live epoch and timestamps.
    pub fn header(&self) -> SliceHeaderInfo {
        let mut header = self.header.clone();
        header.writer_epoch = self.load_epoch();
        header.heartbeat_ns = self.load_u64_atomic(OFF_HEARTBEAT_NS);
        header.updated_ns = self.load_u64_atomic(OFF_UPDATED_NS);
        header
    }

    /// Reject a layout whose hash or capacity differs from this slice.
    pub fn ensure_layout(&self, layout: &LayoutRegistry) -> Result<()> {
        let expected = layout.layout_hash()?;
        let actual = self.header.layout_hash;
        if actual != expected {
            bail!("layout hash mismatch: slice={actual:#x} layout={expected:#x}");
        }
        if self.header.capacity != layout.capacity as u64 {
            bail!(
                "capacity mismatch: slice={} layout={}",
                self.header.capacity,
                layout.capacity
            );
        }
        Ok(())
    }

    /// Atomically read the current whole-vector epoch.
    pub fn load_epoch(&self) -> u64 {
        self.load_u64_atomic(OFF_WRITER_EPOCH)
    }

    /// Return the logical vector length used by guarded operations.
    pub fn active_len(&self) -> usize {
        self.header.active_len as usize
    }

    /// Borrow the complete mapped payload without an epoch guard.
    ///
    /// Callers requiring a coherent copy should prefer [`Self::snapshot_vec`].
    pub fn as_slice(&self) -> &[f64] {
        let ptr = unsafe { self.mmap.as_ptr().add(HEADER_SIZE) as *const f64 };
        unsafe { std::slice::from_raw_parts(ptr, self.header.capacity as usize) }
    }

    /// Copy the active vector under an optimistic epoch guard.
    pub fn snapshot_vec(&self) -> Result<Vec<f64>> {
        retry_epoch(|| {
            let start = self.load_epoch();
            if start % 2 != 0 {
                return Ok(None);
            }
            let values = self.as_slice()[..self.active_len()].to_vec();
            let end = self.load_epoch();
            if start == end && end % 2 == 0 {
                Ok(Some(values))
            } else {
                Ok(None)
            }
        })
    }

    /// Read one slot under an optimistic epoch guard.
    pub fn value_at(&self, slot_id: usize) -> Result<f64> {
        if slot_id >= self.header.capacity as usize {
            bail!(
                "slot_id {slot_id} exceeds capacity {}",
                self.header.capacity
            );
        }
        retry_epoch(|| {
            let start = self.load_epoch();
            if start % 2 != 0 {
                return Ok(None);
            }
            let value = self.as_slice()[slot_id];
            let end = self.load_epoch();
            if start == end && end % 2 == 0 {
                Ok(Some(value))
            } else {
                Ok(None)
            }
        })
    }

    /// Compute a guarded dot product with a layout-compatible slice.
    pub fn dot(&self, other: &Self) -> Result<f64> {
        ensure_compatible_vectors(self, other)?;
        retry_epoch(|| {
            let a0 = self.load_epoch();
            let b0 = other.load_epoch();
            if a0 % 2 != 0 || b0 % 2 != 0 {
                return Ok(None);
            }
            let lhs = &self.as_slice()[..self.active_len()];
            let rhs = &other.as_slice()[..other.active_len()];
            let value = dot(lhs, rhs)?;
            let a1 = self.load_epoch();
            let b1 = other.load_epoch();
            if a0 == a1 && b0 == b1 && a1 % 2 == 0 && b1 % 2 == 0 {
                Ok(Some(value))
            } else {
                Ok(None)
            }
        })
    }

    /// Sum the active vector under an optimistic epoch guard.
    pub fn sum(&self) -> Result<f64> {
        retry_epoch(|| {
            let start = self.load_epoch();
            if start % 2 != 0 {
                return Ok(None);
            }
            let value = self.as_slice()[..self.active_len()].iter().sum();
            let end = self.load_epoch();
            if start == end && end % 2 == 0 {
                Ok(Some(value))
            } else {
                Ok(None)
            }
        })
    }

    /// Sum absolute values under an optimistic epoch guard.
    pub fn sum_abs(&self) -> Result<f64> {
        retry_epoch(|| {
            let start = self.load_epoch();
            if start % 2 != 0 {
                return Ok(None);
            }
            let value = self.as_slice()[..self.active_len()]
                .iter()
                .map(|value| value.abs())
                .sum();
            let end = self.load_epoch();
            if start == end && end % 2 == 0 {
                Ok(Some(value))
            } else {
                Ok(None)
            }
        })
    }

    /// Return up to `limit` finite nonzero slots by descending magnitude.
    pub fn top_abs(&self, limit: usize) -> Result<Vec<(usize, f64)>> {
        let values = self.snapshot_vec()?;
        Ok(top_abs(&values, limit))
    }

    fn load_u64_atomic(&self, offset: usize) -> u64 {
        load_u64_atomic(self.mmap.as_ptr(), offset)
    }
}

/// Writable mapping of a dense `f64` slice.
///
/// The writer owns the publication epoch. Opening the same file through more
/// than one writer is outside the version-1 consistency contract.
pub struct F64SliceWriter {
    path: PathBuf,
    mmap: MmapMut,
    header: SliceHeaderInfo,
}

impl F64SliceWriter {
    /// Create a slice using the layout's active length.
    pub fn create(
        path: impl AsRef<Path>,
        layout: &LayoutRegistry,
        overwrite: bool,
    ) -> Result<Self> {
        let active_len = layout.active_len();
        Self::create_with_active_len(path, layout, active_len, overwrite)
    }

    /// Create a slice with an explicit active prefix.
    pub fn create_with_active_len(
        path: impl AsRef<Path>,
        layout: &LayoutRegistry,
        active_len: u64,
        overwrite: bool,
    ) -> Result<Self> {
        layout.validate()?;
        if active_len > layout.capacity as u64 {
            bail!(
                "active_len {} exceeds layout capacity {}",
                active_len,
                layout.capacity
            );
        }
        let path = path.as_ref().to_path_buf();
        if path.exists() && !overwrite {
            bail!(
                "{} already exists; pass overwrite=true to replace",
                path.display()
            );
        }
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed creating {}", parent.display()))?;
        }

        let capacity = layout.capacity as u64;
        let value_type = ValueType::F64;
        let header = SliceHeaderInfo {
            version: VERSION,
            header_len: HEADER_SIZE as u32,
            value_type,
            flags: FLAG_WRITABLE_OWNER,
            capacity,
            active_len,
            slot_size: value_type.slot_size(),
            layout_hash: layout.layout_hash()?,
            schema_hash: value_type.schema_hash(),
            writer_epoch: 2,
            heartbeat_ns: now_ns(),
            created_ns: now_ns(),
            updated_ns: now_ns(),
        };
        let file_len = header.file_len()?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .with_context(|| format!("failed creating {}", path.display()))?;
        file.set_len(file_len)
            .with_context(|| format!("failed sizing {}", path.display()))?;
        let mut mmap = unsafe { MmapOptions::new().map_mut(&file) }
            .with_context(|| format!("failed mmap writable {}", path.display()))?;
        mmap.fill(0);
        write_header(&mut mmap, &header)?;
        mmap.flush()
            .with_context(|| format!("failed flushing {}", path.display()))?;
        Ok(Self { path, mmap, header })
    }

    /// Open an existing writable `f64` slice.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .with_context(|| format!("failed opening {}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map_mut(&file) }
            .with_context(|| format!("failed mmap writable {}", path.display()))?;
        let header = read_header(&mmap)?;
        validate_f64_header(&header)?;
        validate_file_len(mmap.len(), &header)?;
        Ok(Self { path, mmap, header })
    }

    /// Return the mapped file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read header metadata, refreshing the live epoch and timestamps.
    pub fn header(&self) -> SliceHeaderInfo {
        let mut header = self.header.clone();
        header.writer_epoch = self.load_epoch();
        header.heartbeat_ns = self.load_u64_atomic(OFF_HEARTBEAT_NS);
        header.updated_ns = self.load_u64_atomic(OFF_UPDATED_NS);
        header
    }

    /// Atomically read the current whole-vector epoch.
    pub fn load_epoch(&self) -> u64 {
        self.load_u64_atomic(OFF_WRITER_EPOCH)
    }

    /// Borrow the complete mapped payload without beginning a write epoch.
    pub fn as_slice(&self) -> &[f64] {
        let ptr = unsafe { self.mmap.as_ptr().add(HEADER_SIZE) as *const f64 };
        unsafe { std::slice::from_raw_parts(ptr, self.header.capacity as usize) }
    }

    /// Replace one slot inside its own publication epoch.
    pub fn write_slot(&mut self, slot_id: usize, value: f64) -> Result<()> {
        if slot_id >= self.header.capacity as usize {
            bail!(
                "slot_id {slot_id} exceeds capacity {}",
                self.header.capacity
            );
        }
        self.begin_write();
        {
            let values = self.as_mut_slice();
            values[slot_id] = value;
        }
        self.end_write();
        Ok(())
    }

    /// Mutate the active vector inside one publication epoch.
    pub fn update_vector(&mut self, update: impl FnOnce(&mut [f64])) -> Result<()> {
        self.begin_write();
        {
            let active_len = self.header.active_len as usize;
            let values = &mut self.as_mut_slice()[..active_len];
            update(values);
        }
        self.end_write();
        Ok(())
    }

    /// Flush dirty mapped pages to the backing file.
    pub fn flush(&mut self) -> Result<()> {
        self.mmap
            .flush()
            .with_context(|| format!("failed flushing {}", self.path.display()))
    }

    /// Initiate an asynchronous flush of dirty mapped pages.
    ///
    /// Successful return means the operating system accepted the request, not
    /// that the pages are already durable. Use [`Self::flush`] for a durability
    /// barrier.
    pub fn flush_async(&mut self) -> Result<()> {
        self.mmap
            .flush_async()
            .with_context(|| format!("failed scheduling flush for {}", self.path.display()))
    }

    /// Refresh writer liveness without changing the data timestamp.
    pub fn heartbeat(&mut self) {
        let now = now_ns();
        self.store_u64_atomic(OFF_HEARTBEAT_NS, now);
        self.header.heartbeat_ns = now;
    }

    fn as_mut_slice(&mut self) -> &mut [f64] {
        let ptr = unsafe { self.mmap.as_mut_ptr().add(HEADER_SIZE) as *mut f64 };
        unsafe { std::slice::from_raw_parts_mut(ptr, self.header.capacity as usize) }
    }

    fn begin_write(&self) {
        let current = self.load_epoch();
        let odd = if current % 2 == 0 {
            current.saturating_add(1)
        } else {
            current
        };
        self.store_u64_atomic(OFF_WRITER_EPOCH, odd);
    }

    fn end_write(&mut self) {
        let now = now_ns();
        self.store_u64_atomic(OFF_HEARTBEAT_NS, now);
        self.store_u64_atomic(OFF_UPDATED_NS, now);
        let current = self.load_epoch();
        let even = if current % 2 == 0 {
            current.saturating_add(2)
        } else {
            current.saturating_add(1)
        };
        self.store_u64_atomic(OFF_WRITER_EPOCH, even);
        self.header.writer_epoch = even;
        self.header.heartbeat_ns = now;
        self.header.updated_ns = now;
    }

    fn load_u64_atomic(&self, offset: usize) -> u64 {
        load_u64_atomic(self.mmap.as_ptr(), offset)
    }

    fn store_u64_atomic(&self, offset: usize, value: u64) {
        store_u64_atomic(self.mmap.as_ptr(), offset, value);
    }
}

/// Reader specialized for [`QuoteV1`].
pub type QuoteSliceReader = StructSliceReader<QuoteV1>;
/// Writer specialized for [`QuoteV1`].
pub type QuoteSliceWriter = StructSliceWriter<QuoteV1>;
/// Reader specialized for [`TradeV1`].
pub type TradeSliceReader = StructSliceReader<TradeV1>;
/// Writer specialized for [`TradeV1`].
pub type TradeSliceWriter = StructSliceWriter<TradeV1>;
/// Reader specialized for [`TaqV1`].
pub type TaqSliceReader = StructSliceReader<TaqV1>;
/// Writer specialized for [`TaqV1`].
pub type TaqSliceWriter = StructSliceWriter<TaqV1>;

/// Read-only mapping of a per-slot sequenced fixed-record slice.
pub struct StructSliceReader<T: SliceRecord> {
    path: PathBuf,
    mmap: Mmap,
    header: SliceHeaderInfo,
    _marker: std::marker::PhantomData<T>,
}

impl<T: SliceRecord> StructSliceReader<T> {
    /// Open and validate an existing fixed-record slice.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file =
            File::open(&path).with_context(|| format!("failed opening {}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("failed mmap read-only {}", path.display()))?;
        let header = read_header(&mmap)?;
        validate_struct_header::<T>(&header)?;
        validate_file_len(mmap.len(), &header)?;
        Ok(Self {
            path,
            mmap,
            header,
            _marker: std::marker::PhantomData,
        })
    }

    /// Return the mapped file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read header metadata, refreshing live timestamps.
    pub fn header(&self) -> SliceHeaderInfo {
        let mut header = self.header.clone();
        header.heartbeat_ns = self.load_u64_atomic(OFF_HEARTBEAT_NS);
        header.updated_ns = self.load_u64_atomic(OFF_UPDATED_NS);
        header
    }

    /// Reject a layout whose hash or capacity differs from this slice.
    pub fn ensure_layout(&self, layout: &LayoutRegistry) -> Result<()> {
        let expected = layout.layout_hash()?;
        let actual = self.header.layout_hash;
        if actual != expected {
            bail!("layout hash mismatch: slice={actual:#x} layout={expected:#x}");
        }
        if self.header.capacity != layout.capacity as u64 {
            bail!(
                "capacity mismatch: slice={} layout={}",
                self.header.capacity,
                layout.capacity
            );
        }
        Ok(())
    }

    /// Copy one coherent record using its per-slot sequence counter.
    pub fn value_at(&self, slot_id: usize) -> Result<T> {
        if slot_id >= self.header.capacity as usize {
            bail!(
                "slot_id {slot_id} exceeds capacity {}",
                self.header.capacity
            );
        }
        retry_epoch(|| {
            let slot = self.slot(slot_id);
            let seq1 = slot_seq(slot).load(Ordering::Acquire);
            if seq1 % 2 != 0 {
                return Ok(None);
            }
            std::sync::atomic::fence(Ordering::Acquire);
            let value = unsafe { ptr::read_volatile(slot_value(slot)) };
            std::sync::atomic::fence(Ordering::Acquire);
            let seq2 = slot_seq(slot).load(Ordering::Acquire);
            if seq1 == seq2 && seq2 % 2 == 0 {
                Ok(Some(value))
            } else {
                Ok(None)
            }
        })
    }

    /// Copy each active slot coherently.
    ///
    /// Slots may represent different writer instants; this is not a globally
    /// atomic snapshot of the complete struct slice.
    pub fn snapshot_vec(&self) -> Result<Vec<T>> {
        let mut out = Vec::with_capacity(self.header.active_len as usize);
        for slot_id in 0..self.header.active_len as usize {
            out.push(self.value_at(slot_id)?);
        }
        Ok(out)
    }

    fn slot(&self, slot_id: usize) -> *const SequencedRecord<T> {
        let ptr = unsafe { self.mmap.as_ptr().add(HEADER_SIZE) as *const SequencedRecord<T> };
        unsafe { ptr.add(slot_id) }
    }

    fn load_u64_atomic(&self, offset: usize) -> u64 {
        load_u64_atomic(self.mmap.as_ptr(), offset)
    }
}

/// Writable mapping of a per-slot sequenced fixed-record slice.
pub struct StructSliceWriter<T: SliceRecord> {
    path: PathBuf,
    mmap: MmapMut,
    header: SliceHeaderInfo,
    _marker: std::marker::PhantomData<T>,
}

impl<T: SliceRecord> StructSliceWriter<T> {
    /// Create a fixed-record slice using the layout's active length.
    pub fn create(
        path: impl AsRef<Path>,
        layout: &LayoutRegistry,
        overwrite: bool,
    ) -> Result<Self> {
        let active_len = layout.active_len();
        Self::create_with_active_len(path, layout, active_len, overwrite)
    }

    /// Create a fixed-record slice with an explicit active prefix.
    pub fn create_with_active_len(
        path: impl AsRef<Path>,
        layout: &LayoutRegistry,
        active_len: u64,
        overwrite: bool,
    ) -> Result<Self> {
        layout.validate()?;
        if active_len > layout.capacity as u64 {
            bail!(
                "active_len {} exceeds layout capacity {}",
                active_len,
                layout.capacity
            );
        }
        let path = path.as_ref().to_path_buf();
        if path.exists() && !overwrite {
            bail!(
                "{} already exists; pass overwrite=true to replace",
                path.display()
            );
        }
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed creating {}", parent.display()))?;
        }

        let value_type = T::VALUE_TYPE;
        let header = SliceHeaderInfo {
            version: VERSION,
            header_len: HEADER_SIZE as u32,
            value_type,
            flags: FLAG_WRITABLE_OWNER,
            capacity: layout.capacity as u64,
            active_len,
            slot_size: value_type.slot_size(),
            layout_hash: layout.layout_hash()?,
            schema_hash: value_type.schema_hash(),
            writer_epoch: 2,
            heartbeat_ns: now_ns(),
            created_ns: now_ns(),
            updated_ns: now_ns(),
        };
        let file_len = header.file_len()?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .with_context(|| format!("failed creating {}", path.display()))?;
        file.set_len(file_len)
            .with_context(|| format!("failed sizing {}", path.display()))?;
        let mut mmap = unsafe { MmapOptions::new().map_mut(&file) }
            .with_context(|| format!("failed mmap writable {}", path.display()))?;
        mmap.fill(0);
        write_header(&mut mmap, &header)?;
        mmap.flush()
            .with_context(|| format!("failed flushing {}", path.display()))?;
        Ok(Self {
            path,
            mmap,
            header,
            _marker: std::marker::PhantomData,
        })
    }

    /// Open an existing writable fixed-record slice.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .with_context(|| format!("failed opening {}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map_mut(&file) }
            .with_context(|| format!("failed mmap writable {}", path.display()))?;
        let header = read_header(&mmap)?;
        validate_struct_header::<T>(&header)?;
        validate_file_len(mmap.len(), &header)?;
        Ok(Self {
            path,
            mmap,
            header,
            _marker: std::marker::PhantomData,
        })
    }

    /// Return the mapped file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Return the writer's current header metadata.
    pub fn header(&self) -> SliceHeaderInfo {
        self.header.clone()
    }

    /// Publish one coherent fixed record through its sequence counter.
    pub fn write_slot(&mut self, slot_id: usize, value: T) -> Result<()> {
        if slot_id >= self.header.capacity as usize {
            bail!(
                "slot_id {slot_id} exceeds capacity {}",
                self.header.capacity
            );
        }
        let slot = self.slot(slot_id);
        let seq = slot_seq(slot).load(Ordering::Acquire);
        let odd = if seq % 2 == 0 {
            seq.saturating_add(1)
        } else {
            seq
        };
        slot_seq(slot).store(odd, Ordering::SeqCst);
        std::sync::atomic::fence(Ordering::SeqCst);
        unsafe { ptr::write_volatile(slot_value_mut(slot), value) };
        std::sync::atomic::fence(Ordering::Release);
        slot_seq(slot).store(odd.saturating_add(1), Ordering::Release);
        self.touch();
        Ok(())
    }

    /// Flush dirty mapped pages to the backing file.
    pub fn flush(&mut self) -> Result<()> {
        self.mmap
            .flush()
            .with_context(|| format!("failed flushing {}", self.path.display()))
    }

    /// Refresh writer liveness without changing the data timestamp.
    pub fn heartbeat(&mut self) {
        let now = now_ns();
        store_u64_atomic(self.mmap.as_ptr(), OFF_HEARTBEAT_NS, now);
        self.header.heartbeat_ns = now;
    }

    fn touch(&mut self) {
        let now = now_ns();
        store_u64_atomic(self.mmap.as_ptr(), OFF_HEARTBEAT_NS, now);
        store_u64_atomic(self.mmap.as_ptr(), OFF_UPDATED_NS, now);
        self.header.heartbeat_ns = now;
        self.header.updated_ns = now;
    }

    fn slot(&mut self, slot_id: usize) -> *mut SequencedRecord<T> {
        let ptr = unsafe { self.mmap.as_mut_ptr().add(HEADER_SIZE) as *mut SequencedRecord<T> };
        unsafe { ptr.add(slot_id) }
    }
}

/// Compute the dot product of equally sized in-memory vectors.
pub fn dot(lhs: &[f64], rhs: &[f64]) -> Result<f64> {
    if lhs.len() != rhs.len() {
        bail!("dot length mismatch: lhs={} rhs={}", lhs.len(), rhs.len());
    }
    Ok(lhs.iter().zip(rhs).map(|(a, b)| a * b).sum())
}

/// Sum absolute values in an in-memory vector.
pub fn sum_abs(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).sum()
}

/// Return up to `limit` finite nonzero values by descending magnitude.
///
/// Selection is linear in the input length, followed by a sort of only the
/// selected prefix. Equal magnitudes are ordered by ascending slot so results
/// remain deterministic.
pub fn top_abs(values: &[f64], limit: usize) -> Vec<(usize, f64)> {
    if limit == 0 {
        return Vec::new();
    }
    let mut indexed = values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite() && *value != 0.0)
        .collect::<Vec<_>>();
    let compare = |(left_slot, left): &(usize, f64), (right_slot, right): &(usize, f64)| {
        right
            .abs()
            .total_cmp(&left.abs())
            .then_with(|| left_slot.cmp(right_slot))
    };
    if limit < indexed.len() {
        indexed.select_nth_unstable_by(limit, compare);
        indexed.truncate(limit);
    }
    indexed.sort_unstable_by(compare);
    indexed
}

/// Trim and uppercase a nonempty key without embedded whitespace.
pub fn normalize_symbol(raw: &str) -> Result<String> {
    let symbol = raw.trim().to_ascii_uppercase();
    if symbol.is_empty() {
        bail!("symbol cannot be empty");
    }
    if symbol
        .chars()
        .any(|ch| ch.is_control() || ch.is_whitespace())
    {
        bail!("symbol contains unsupported whitespace/control characters: {raw:?}");
    }
    Ok(symbol)
}

/// Normalize known financial namespace aliases or validate a custom namespace.
pub fn normalize_asset_class(raw: &str) -> Result<String> {
    let value = raw.trim().to_ascii_lowercase();
    if value.is_empty() {
        bail!("asset_class cannot be empty");
    }
    let normalized = match value.as_str() {
        "eq" | "equity" | "equities" | "stock" | "stocks" => "eq".to_string(),
        "crypto" | "cryptocurrency" | "coin" | "coins" => "crypto".to_string(),
        "option" | "options" | "opt" | "op" => "option".to_string(),
        "future" | "futures" | "fut" | "fu" => "future".to_string(),
        "fx" | "forex" | "currency" | "currencies" => "fx".to_string(),
        "index" | "indices" | "idx" | "ix" => "index".to_string(),
        _ => value,
    };
    if normalized
        .chars()
        .any(|ch| !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '-'))
    {
        bail!("asset_class contains unsupported characters: {raw:?}");
    }
    Ok(normalized)
}

/// Build a default stable identifier from namespace and normalized symbol.
pub fn default_instrument_id(
    asset_class: &str,
    symbol: &str,
    instrument_prefix: Option<&str>,
) -> Result<String> {
    let asset_class = normalize_asset_class(asset_class)?;
    let symbol = normalize_symbol(symbol)?;
    if let Some(prefix) = instrument_prefix {
        let prefix = prefix.trim();
        if prefix.is_empty() {
            bail!("instrument prefix cannot be empty");
        }
        return Ok(format!("{prefix}{symbol}"));
    }
    let instrument_id = match asset_class.as_str() {
        "eq" => format!("eq:us:{symbol}"),
        "crypto" => format!("crypto:{symbol}"),
        "option" => format!("option:us:{symbol}"),
        "future" => format!("future:{symbol}"),
        "fx" => format!("fx:{symbol}"),
        "index" => format!("index:{symbol}"),
        other => format!("{other}:{symbol}"),
    };
    Ok(instrument_id)
}

/// Validate the characters admitted in a catalog slice name.
pub fn validate_slice_name(name: &str) -> Result<()> {
    if name.trim() != name || name.is_empty() {
        bail!("slice name cannot be empty or padded");
    }
    if name
        .chars()
        .any(|ch| !(ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-' | '/' | ':')))
    {
        bail!("slice name contains unsupported characters: {name:?}");
    }
    Ok(())
}

/// Decode and validate an existing slice header without mapping its payload.
pub fn read_slice_header(path: impl AsRef<Path>) -> Result<SliceHeaderInfo> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("failed opening {}", path.display()))?;
    let mmap = unsafe { MmapOptions::new().map(&file) }
        .with_context(|| format!("failed mmap read-only {}", path.display()))?;
    let header = read_header(&mmap)?;
    validate_file_len(mmap.len(), &header)?;
    Ok(header)
}

fn ensure_compatible_vectors(lhs: &F64SliceReader, rhs: &F64SliceReader) -> Result<()> {
    if lhs.header.value_type != rhs.header.value_type {
        bail!(
            "value type mismatch: lhs={:?} rhs={:?}",
            lhs.header.value_type,
            rhs.header.value_type
        );
    }
    if lhs.header.layout_hash != rhs.header.layout_hash {
        bail!(
            "layout hash mismatch: lhs={:#x} rhs={:#x}",
            lhs.header.layout_hash,
            rhs.header.layout_hash
        );
    }
    if lhs.header.schema_hash != rhs.header.schema_hash {
        bail!(
            "schema hash mismatch: lhs={:#x} rhs={:#x}",
            lhs.header.schema_hash,
            rhs.header.schema_hash
        );
    }
    if lhs.header.active_len != rhs.header.active_len {
        bail!(
            "active_len mismatch: lhs={} rhs={}",
            lhs.header.active_len,
            rhs.header.active_len
        );
    }
    Ok(())
}

fn retry_epoch<T>(mut f: impl FnMut() -> Result<Option<T>>) -> Result<T> {
    for _ in 0..1000 {
        if let Some(value) = f()? {
            return Ok(value);
        }
        thread::sleep(Duration::from_micros(50));
    }
    bail!("could not read a stable slice snapshot after 1000 attempts")
}

fn read_header(bytes: &[u8]) -> Result<SliceHeaderInfo> {
    if bytes.len() < HEADER_SIZE {
        bail!("slice is too small: {} bytes", bytes.len());
    }
    if &bytes[OFF_MAGIC..OFF_MAGIC + MAGIC.len()] != MAGIC {
        bail!("invalid slice magic");
    }
    let version = read_u32(bytes, OFF_VERSION)?;
    if version != VERSION {
        bail!("unsupported slice version {version}; expected {VERSION}");
    }
    let header = SliceHeaderInfo {
        version,
        header_len: read_u32(bytes, OFF_HEADER_LEN)?,
        value_type: ValueType::from_u32(read_u32(bytes, OFF_VALUE_TYPE)?)?,
        flags: read_u32(bytes, OFF_FLAGS)?,
        capacity: read_u64(bytes, OFF_CAPACITY)?,
        active_len: read_u64(bytes, OFF_ACTIVE_LEN)?,
        slot_size: read_u64(bytes, OFF_SLOT_SIZE)?,
        layout_hash: read_u64(bytes, OFF_LAYOUT_HASH)?,
        schema_hash: read_u64(bytes, OFF_SCHEMA_HASH)?,
        writer_epoch: load_u64_atomic(bytes.as_ptr(), OFF_WRITER_EPOCH),
        heartbeat_ns: load_u64_atomic(bytes.as_ptr(), OFF_HEARTBEAT_NS),
        created_ns: read_u64(bytes, OFF_CREATED_NS)?,
        updated_ns: load_u64_atomic(bytes.as_ptr(), OFF_UPDATED_NS),
    };
    if header.header_len as usize != HEADER_SIZE {
        bail!(
            "unsupported header length {}; expected {}",
            header.header_len,
            HEADER_SIZE
        );
    }
    if header.active_len > header.capacity {
        bail!(
            "active_len {} exceeds capacity {}",
            header.active_len,
            header.capacity
        );
    }
    Ok(header)
}

fn write_header(bytes: &mut [u8], header: &SliceHeaderInfo) -> Result<()> {
    if bytes.len() < HEADER_SIZE {
        bail!("target buffer too small for slice header");
    }
    bytes[..HEADER_SIZE].fill(0);
    bytes[OFF_MAGIC..OFF_MAGIC + MAGIC.len()].copy_from_slice(MAGIC);
    write_u32(bytes, OFF_VERSION, header.version)?;
    write_u32(bytes, OFF_HEADER_LEN, header.header_len)?;
    write_u32(bytes, OFF_VALUE_TYPE, header.value_type as u32)?;
    write_u32(bytes, OFF_FLAGS, header.flags)?;
    write_u64(bytes, OFF_CAPACITY, header.capacity)?;
    write_u64(bytes, OFF_ACTIVE_LEN, header.active_len)?;
    write_u64(bytes, OFF_SLOT_SIZE, header.slot_size)?;
    write_u64(bytes, OFF_LAYOUT_HASH, header.layout_hash)?;
    write_u64(bytes, OFF_SCHEMA_HASH, header.schema_hash)?;
    write_u64(bytes, OFF_WRITER_EPOCH, header.writer_epoch)?;
    write_u64(bytes, OFF_HEARTBEAT_NS, header.heartbeat_ns)?;
    write_u64(bytes, OFF_CREATED_NS, header.created_ns)?;
    write_u64(bytes, OFF_UPDATED_NS, header.updated_ns)?;
    Ok(())
}

fn validate_f64_header(header: &SliceHeaderInfo) -> Result<()> {
    if header.value_type != ValueType::F64 {
        bail!("expected f64 slice, got {:?}", header.value_type);
    }
    if header.slot_size != mem::size_of::<f64>() as u64 {
        bail!("invalid f64 slot size {}", header.slot_size);
    }
    if header.schema_hash != ValueType::F64.schema_hash() {
        bail!(
            "schema hash mismatch: slice={:#x} expected={:#x}",
            header.schema_hash,
            ValueType::F64.schema_hash()
        );
    }
    Ok(())
}

fn validate_struct_header<T: SliceRecord>(header: &SliceHeaderInfo) -> Result<()> {
    if header.value_type != T::VALUE_TYPE {
        bail!(
            "expected {:?} slice, got {:?}",
            T::VALUE_TYPE,
            header.value_type
        );
    }
    if header.slot_size != T::VALUE_TYPE.slot_size() {
        bail!(
            "invalid {:?} slot size {}; expected {}",
            T::VALUE_TYPE,
            header.slot_size,
            T::VALUE_TYPE.slot_size()
        );
    }
    if header.schema_hash != T::VALUE_TYPE.schema_hash() {
        bail!(
            "schema hash mismatch: slice={:#x} expected={:#x}",
            header.schema_hash,
            T::VALUE_TYPE.schema_hash()
        );
    }
    Ok(())
}

fn validate_file_len(actual: usize, header: &SliceHeaderInfo) -> Result<()> {
    let expected = header.file_len()?;
    if actual as u64 != expected {
        bail!("slice file length mismatch: actual={actual} expected={expected}");
    }
    Ok(())
}

fn slot_seq<T: SliceRecord>(slot: *const SequencedRecord<T>) -> &'static AtomicU64 {
    unsafe { &*(slot as *const AtomicU64) }
}

fn slot_value<T: SliceRecord>(slot: *const SequencedRecord<T>) -> *const T {
    unsafe { std::ptr::addr_of!((*slot).value) }
}

fn slot_value_mut<T: SliceRecord>(slot: *mut SequencedRecord<T>) -> *mut T {
    unsafe { std::ptr::addr_of_mut!((*slot).value) }
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let end = offset + 4;
    let chunk = bytes
        .get(offset..end)
        .ok_or_else(|| anyhow!("offset {offset} out of bounds"))?;
    Ok(u32::from_le_bytes(chunk.try_into().unwrap()))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let end = offset + 8;
    let chunk = bytes
        .get(offset..end)
        .ok_or_else(|| anyhow!("offset {offset} out of bounds"))?;
    Ok(u64::from_le_bytes(chunk.try_into().unwrap()))
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) -> Result<()> {
    let end = offset + 4;
    let chunk = bytes
        .get_mut(offset..end)
        .ok_or_else(|| anyhow!("offset {offset} out of bounds"))?;
    chunk.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) -> Result<()> {
    let end = offset + 8;
    let chunk = bytes
        .get_mut(offset..end)
        .ok_or_else(|| anyhow!("offset {offset} out of bounds"))?;
    chunk.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn load_u64_atomic(base: *const u8, offset: usize) -> u64 {
    debug_assert_eq!(offset % mem::align_of::<AtomicU64>(), 0);
    let ptr = unsafe { base.add(offset) as *const AtomicU64 };
    unsafe { (*ptr).load(Ordering::Acquire) }
}

fn store_u64_atomic(base: *const u8, offset: usize, value: u64) {
    debug_assert_eq!(offset % mem::align_of::<AtomicU64>(), 0);
    let ptr = unsafe { base.add(offset) as *const AtomicU64 };
    unsafe { (*ptr).store(value, Ordering::Release) };
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64
}

/// Format a nanosecond Unix timestamp as seconds plus zero-padded nanoseconds.
pub fn format_ns_since_epoch(ns: u64) -> String {
    if ns == 0 {
        return "0".to_string();
    }
    let secs = ns / 1_000_000_000;
    let sub = ns % 1_000_000_000;
    format!("{secs}.{sub:09}s")
}
