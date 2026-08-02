//! Projection of coherent in-process snapshots into independently readable slices.
//!
//! The publisher owns layout and catalog creation as well as the writer for
//! each selected node. Cross-slice atomicity remains outside this module's
//! contract.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use hypercube_slice::{
    validate_slice_name, F64SliceWriter, LayoutRegistry, SliceCatalog, SliceCatalogEntry, ValueType,
};

use crate::Snapshot;

/// Publishes selected node cross-sections as memory-mapped `f64` slices.
///
/// A publisher owns the writer side of each slice. Any number of other
/// processes may open the resulting files with
/// [`hypercube_slice::F64SliceReader`].
pub struct SlicePublisher {
    root: PathBuf,
    layout: LayoutRegistry,
    nodes: BTreeMap<String, PublishedNode>,
    slots: HashMap<String, usize>,
}

struct PublishedNode {
    writer: F64SliceWriter,
    buffer: Vec<f64>,
}

/// Persistence behavior after a snapshot becomes visible in mapped memory.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PublishDurability {
    /// Return after the writer epochs make the new values visible to readers.
    MemoryMapped,
    /// Ask the operating system to begin flushing dirty pages without waiting.
    Async,
    /// Wait until every configured slice reports its dirty pages as durable.
    #[default]
    Durable,
}

impl SlicePublisher {
    /// Create a publisher without replacing any existing layout, catalog, or
    /// slice files beneath `root`.
    pub fn create(
        root: impl AsRef<Path>,
        layout_id: impl Into<String>,
        entity_keys: &[String],
        node_ids: &[String],
    ) -> Result<Self> {
        Self::create_inner(root, layout_id, entity_keys, node_ids, false)
    }

    /// Create a publisher and replace the files for the declared layout and
    /// nodes when they already exist.
    pub fn create_overwrite(
        root: impl AsRef<Path>,
        layout_id: impl Into<String>,
        entity_keys: &[String],
        node_ids: &[String],
    ) -> Result<Self> {
        Self::create_inner(root, layout_id, entity_keys, node_ids, true)
    }

    fn create_inner(
        root: impl AsRef<Path>,
        layout_id: impl Into<String>,
        entity_keys: &[String],
        node_ids: &[String],
        overwrite: bool,
    ) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let capacity =
            u32::try_from(entity_keys.len()).context("too many entities for a layout")?;
        let layout = LayoutRegistry::from_entities(layout_id, "synthetic", capacity, entity_keys)?;
        let layout_path = root.join("layout.json");
        let mut unique_nodes = BTreeSet::new();
        let mut paths = BTreeMap::new();
        for node in node_ids {
            validate_slice_name(node)?;
            if !unique_nodes.insert(node.as_str()) {
                return Err(anyhow!("duplicate slice node {node}"));
            }
            let filename = format!("{}.slice", safe_filename(node));
            let path = root.join("slices").join(filename);
            paths.insert(node.clone(), path);
        }
        if !overwrite {
            let catalog_path = root.join("catalog.json");
            let existing = std::iter::once(&layout_path)
                .chain(std::iter::once(&catalog_path))
                .chain(paths.values())
                .find(|path| path.exists());
            if let Some(path) = existing {
                return Err(anyhow!(
                    "{} already exists; use create_overwrite to replace it",
                    path.display()
                ));
            }
        }

        layout.save_pretty(&layout_path)?;
        let mut nodes = BTreeMap::new();
        let mut catalog = SliceCatalog::default();
        let layout_hash = layout.layout_hash()?;
        for (node, path) in paths {
            let writer = F64SliceWriter::create(&path, &layout, overwrite)?;
            catalog.upsert(SliceCatalogEntry {
                name: node.clone(),
                asset_class: layout.asset_class.clone(),
                layout_id: layout.layout_id.clone(),
                layout_hash,
                value_type: ValueType::F64,
                path: path.to_string_lossy().into_owned(),
                role: "node_output".to_owned(),
                description: Some(format!("Hypercube output for node {node}")),
            })?;
            nodes.insert(
                node,
                PublishedNode {
                    writer,
                    buffer: vec![f64::NAN; layout.active_len() as usize],
                },
            );
        }
        catalog.save_pretty(root.join("catalog.json"))?;
        let slots = entity_keys
            .iter()
            .enumerate()
            .map(|(slot, key)| (key.clone(), slot))
            .collect();
        Ok(Self {
            root,
            layout,
            nodes,
            slots,
        })
    }

    /// Return the directory containing the layout, catalog, and slice files.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Return the entity layout shared by every configured output slice.
    pub fn layout(&self) -> &LayoutRegistry {
        &self.layout
    }

    /// Publish every configured node from one coherent engine snapshot.
    ///
    /// Each output slice becomes coherent independently. This method does not
    /// claim atomic publication across the complete set of files. For backward
    /// compatibility this method performs a durability barrier after updating
    /// all slices. Low-latency views should use
    /// [`Self::publish_with_durability`] with
    /// [`PublishDurability::MemoryMapped`].
    pub fn publish(&mut self, snapshot: &Snapshot) -> Result<()> {
        self.publish_with_durability(snapshot, PublishDurability::Durable)
    }

    /// Publish a snapshot using an explicit memory-versus-durability policy.
    ///
    /// Projection is performed in one pass over the snapshot into buffers
    /// allocated when the publisher is created. Writer epochs are completed
    /// for every node before the selected flush policy is applied.
    pub fn publish_with_durability(
        &mut self,
        snapshot: &Snapshot,
        durability: PublishDurability,
    ) -> Result<()> {
        for published in self.nodes.values_mut() {
            published.buffer.fill(f64::NAN);
        }
        for value in &snapshot.values {
            let slot = match self.slots.get(&value.key).copied() {
                Some(slot) => slot,
                None => {
                    let slot =
                        self.layout.slot_for_entity(&value.key)?.ok_or_else(|| {
                            anyhow!("snapshot contains unknown entity {}", value.key)
                        })? as usize;
                    self.slots.insert(value.key.clone(), slot);
                    slot
                }
            };
            let published = self.nodes.get_mut(&value.node).ok_or_else(|| {
                anyhow!("snapshot contains unconfigured slice node {}", value.node)
            })?;
            if slot >= published.buffer.len() {
                return Err(anyhow!(
                    "snapshot entity {} resolves outside the active layout",
                    value.key
                ));
            }
            published.buffer[slot] = value.value;
        }

        for published in self.nodes.values_mut() {
            let buffer = &published.buffer;
            published
                .writer
                .update_vector(|output| output.copy_from_slice(buffer))?;
        }
        match durability {
            PublishDurability::MemoryMapped => Ok(()),
            PublishDurability::Async => self.flush_async(),
            PublishDurability::Durable => self.flush(),
        }
    }

    /// Wait until dirty pages for every configured slice are durable.
    pub fn flush(&mut self) -> Result<()> {
        for (node, published) in &mut self.nodes {
            published
                .writer
                .flush()
                .with_context(|| format!("failed publishing node {node}"))?;
        }
        Ok(())
    }

    /// Initiate asynchronous page flushes for every configured slice.
    pub fn flush_async(&mut self) -> Result<()> {
        for (node, published) in &mut self.nodes {
            published
                .writer
                .flush_async()
                .with_context(|| format!("failed scheduling publication for node {node}"))?;
        }
        Ok(())
    }
}

fn safe_filename(node: &str) -> String {
    node.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use hypercube_slice::F64SliceReader;

    use super::*;
    use crate::{CellValue, ExecutionMode, NodeStatus, PublishDurability, Snapshot};

    #[test]
    fn publishes_aligned_node_vectors() {
        let temp = tempfile::tempdir().unwrap();
        let entities = vec!["A".to_owned(), "B".to_owned()];
        let nodes = vec!["signal".to_owned()];
        let mut publisher =
            SlicePublisher::create(temp.path(), "demo-v1", &entities, &nodes).unwrap();
        publisher
            .publish_with_durability(
                &Snapshot {
                    generation: 1,
                    observed_at_ms: 10,
                    mode: ExecutionMode::Live,
                    entity_count: 2,
                    values: vec![
                        CellValue {
                            node: "signal".to_owned(),
                            key: "A".to_owned(),
                            value: 1.5,
                            observed_at_ms: 10,
                        },
                        CellValue {
                            node: "signal".to_owned(),
                            key: "B".to_owned(),
                            value: -0.5,
                            observed_at_ms: 10,
                        },
                    ],
                    statuses: vec![NodeStatus {
                        node: "signal".to_owned(),
                        values: 2,
                        missing: 0,
                        compute_micros: 1,
                    }],
                },
                PublishDurability::MemoryMapped,
            )
            .unwrap();

        let reader = F64SliceReader::open(temp.path().join("slices/signal.slice")).unwrap();
        assert_eq!(reader.snapshot_vec().unwrap(), vec![1.5, -0.5]);
        publisher.flush_async().unwrap();
        assert!(SlicePublisher::create(temp.path(), "demo-v1", &entities, &nodes).is_err());
    }
}
