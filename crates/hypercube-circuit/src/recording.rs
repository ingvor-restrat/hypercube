use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use hypercube::{NodeKind, Update};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{ThresholdTriggerSpec, TriggerConfigError, TriggerFrame, TriggerProcessor};

/// Current JSON Lines recording schema version.
pub const RECORDING_FORMAT_VERSION: u32 = 1;

/// Immutable metadata required to interpret and reproduce a recording.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordingManifest {
    /// Recording schema version.
    pub format_version: u32,
    /// Stable identity of the original live or synthetic run.
    pub run_id: String,
    /// Application build or source revision that produced the recording.
    pub build_id: String,
    /// Identity of configuration outside the serialized Hypercube node graph.
    pub config_hash: String,
    /// Entity-layout identity used by published vectors.
    pub layout_hash: String,
    /// Stateful trigger configuration used to produce expected transitions.
    pub triggers: Vec<ThresholdTriggerSpec>,
    /// Transport or deployment provenance such as Aeron recording identities.
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

impl RecordingManifest {
    /// Construct and validate a version-1 recording manifest.
    pub fn new(
        run_id: impl Into<String>,
        build_id: impl Into<String>,
        config_hash: impl Into<String>,
        layout_hash: impl Into<String>,
        triggers: Vec<ThresholdTriggerSpec>,
    ) -> Result<Self, RecordingError> {
        let manifest = Self {
            format_version: RECORDING_FORMAT_VERSION,
            run_id: run_id.into(),
            build_id: build_id.into(),
            config_hash: config_hash.into(),
            layout_hash: layout_hash.into(),
            triggers,
            metadata: BTreeMap::new(),
        };
        manifest.validate()?;
        Ok(manifest)
    }

    /// Add one nonempty provenance entry and return the updated manifest.
    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<Self, RecordingError> {
        self.metadata.insert(key.into(), value.into());
        self.validate()?;
        Ok(self)
    }

    fn validate(&self) -> Result<(), RecordingError> {
        if self.format_version != RECORDING_FORMAT_VERSION {
            return Err(RecordingError::UnsupportedFormat {
                actual: self.format_version,
                supported: RECORDING_FORMAT_VERSION,
            });
        }
        for (name, value) in [
            ("run_id", self.run_id.as_str()),
            ("build_id", self.build_id.as_str()),
            ("config_hash", self.config_hash.as_str()),
            ("layout_hash", self.layout_hash.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(RecordingError::EmptyManifestField(name));
            }
        }
        for (key, value) in &self.metadata {
            if key.trim().is_empty() || value.trim().is_empty() {
                return Err(RecordingError::InvalidMetadata { key: key.clone() });
            }
        }
        TriggerProcessor::new(self.triggers.clone())?;
        Ok(())
    }
}

/// One accepted Hypercube update and its expected stateful output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationRecord {
    /// Zero-based sequence assigned by the generation circuit.
    pub circuit_sequence: u64,
    /// Complete input generation accepted by the Hypercube engine.
    pub update: Update,
    /// Last accepted position for each upstream stream at this generation.
    #[serde(default)]
    pub source_positions: BTreeMap<String, i64>,
    /// Semantic snapshot digest and transitions produced in the original run.
    pub expected: TriggerFrame,
}

impl GenerationRecord {
    fn validate(&self) -> Result<(), RecordingError> {
        validate_recordable_update(&self.update)?;
        if self.update.generation != self.expected.generation {
            return Err(RecordingError::ExpectedGenerationMismatch {
                update: self.update.generation,
                expected: self.expected.generation,
            });
        }
        for transition in &self.expected.transitions {
            if transition.value.is_some_and(|value| !value.is_finite()) {
                return Err(RecordingError::NonFiniteValue {
                    location: format!(
                        "transition {} entity {} generation {}",
                        transition.trigger, transition.key, transition.generation
                    ),
                });
            }
        }
        Ok(())
    }
}

pub(crate) fn validate_recordable_update(update: &Update) -> Result<(), RecordingError> {
    for row in &update.rows {
        for (field, value) in &row.fields {
            if !value.is_finite() {
                return Err(RecordingError::NonFiniteValue {
                    location: format!("row {} field {field}", row.key),
                });
            }
        }
    }
    for node in &update.nodes {
        if let NodeKind::Linear { inputs, .. } = &node.kind {
            for input in inputs {
                if !input.weight.is_finite() {
                    return Err(RecordingError::NonFiniteValue {
                        location: format!("node {} input {} weight", node.id, input.node),
                    });
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
#[serde(tag = "record_type", rename_all = "snake_case")]
enum RecordingLine {
    Manifest { manifest: RecordingManifest },
    Generation { record: Box<GenerationRecord> },
}

#[derive(Serialize)]
#[serde(tag = "record_type", rename_all = "snake_case")]
enum RecordingLineRef<'a> {
    Manifest { manifest: &'a RecordingManifest },
    Generation { record: &'a GenerationRecord },
}

/// Streaming recording format or validation error.
#[derive(Debug, Error)]
pub enum RecordingError {
    /// Filesystem or stream operation failed.
    #[error(transparent)]
    Io(#[from] io::Error),
    /// One JSON record could not be encoded or decoded.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Recording format was newer or otherwise unsupported.
    #[error("unsupported recording format {actual}; supported version is {supported}")]
    UnsupportedFormat {
        /// Version read from the manifest.
        actual: u32,
        /// Version supported by this crate.
        supported: u32,
    },
    /// Required manifest metadata was blank.
    #[error("recording manifest field {0} cannot be empty")]
    EmptyManifestField(&'static str),
    /// A provenance key or value was blank.
    #[error("recording metadata key and value must be nonempty: {key:?}")]
    InvalidMetadata {
        /// Invalid metadata key, which may itself be blank.
        key: String,
    },
    /// JSON cannot preserve a NaN or infinite numeric input.
    #[error("recording value at {location} must be finite; omit missing primitive fields")]
    NonFiniteValue {
        /// Coordinate containing the unsupported value.
        location: String,
    },
    /// Stateful trigger configuration was invalid.
    #[error(transparent)]
    Trigger(#[from] TriggerConfigError),
    /// The first nonempty JSON line was not a manifest.
    #[error("recording must begin with exactly one manifest")]
    MissingManifest,
    /// A second manifest appeared after generation data began.
    #[error("recording contains more than one manifest")]
    DuplicateManifest,
    /// Generation records were not contiguous in circuit order.
    #[error("expected circuit sequence {expected}, found {actual}")]
    SequenceMismatch {
        /// Required next sequence.
        expected: u64,
        /// Sequence found in the recording.
        actual: u64,
    },
    /// Hypercube generations did not advance.
    #[error("generation {actual} is not newer than {previous}")]
    NonIncreasingGeneration {
        /// Previously recorded generation.
        previous: u64,
        /// Invalid next generation.
        actual: u64,
    },
    /// Expected output named a different generation than its input.
    #[error("recorded update generation {update} differs from expected output {expected}")]
    ExpectedGenerationMismatch {
        /// Generation carried by the update.
        update: u64,
        /// Generation carried by the expected output.
        expected: u64,
    },
    /// A blank recording contained no manifest.
    #[error("recording is empty")]
    EmptyRecording,
}

/// Streaming writer for the versioned JSON Lines replay format.
pub struct RecordingWriter<W: Write> {
    writer: W,
    next_sequence: u64,
    last_generation: Option<u64>,
}

impl<W: Write> RecordingWriter<W> {
    /// Begin a recording by writing its manifest.
    pub fn new(mut writer: W, manifest: RecordingManifest) -> Result<Self, RecordingError> {
        manifest.validate()?;
        write_line(
            &mut writer,
            &RecordingLineRef::Manifest {
                manifest: &manifest,
            },
        )?;
        Ok(Self {
            writer,
            next_sequence: 0,
            last_generation: None,
        })
    }

    /// Append one validated generation record.
    pub fn write_generation(&mut self, record: &GenerationRecord) -> Result<(), RecordingError> {
        record.validate()?;
        if record.circuit_sequence != self.next_sequence {
            return Err(RecordingError::SequenceMismatch {
                expected: self.next_sequence,
                actual: record.circuit_sequence,
            });
        }
        if let Some(previous) = self.last_generation {
            if record.update.generation <= previous {
                return Err(RecordingError::NonIncreasingGeneration {
                    previous,
                    actual: record.update.generation,
                });
            }
        }
        write_line(&mut self.writer, &RecordingLineRef::Generation { record })?;
        self.next_sequence += 1;
        self.last_generation = Some(record.update.generation);
        Ok(())
    }

    /// Flush buffered recording bytes to the underlying writer.
    pub fn flush(&mut self) -> Result<(), RecordingError> {
        self.writer.flush()?;
        Ok(())
    }

    /// Flush and return the underlying writer.
    pub fn into_inner(mut self) -> Result<W, RecordingError> {
        self.flush()?;
        Ok(self.writer)
    }
}

impl RecordingWriter<BufWriter<File>> {
    /// Create or truncate a recording file and write its manifest.
    pub fn create(
        path: impl AsRef<Path>,
        manifest: RecordingManifest,
    ) -> Result<Self, RecordingError> {
        let file = File::create(path)?;
        Self::new(BufWriter::new(file), manifest)
    }
}

/// Streaming reader that validates recording order as generations are read.
pub struct RecordingReader<R: BufRead> {
    reader: R,
    manifest: RecordingManifest,
    line_number: usize,
    next_sequence: u64,
    last_generation: Option<u64>,
    finished: bool,
}

impl<R: BufRead> RecordingReader<R> {
    /// Read and validate the manifest at the start of a recording.
    pub fn new(mut reader: R) -> Result<Self, RecordingError> {
        let mut line_number = 0;
        let Some(line) = read_nonempty_line(&mut reader, &mut line_number)? else {
            return Err(RecordingError::EmptyRecording);
        };
        let manifest = match serde_json::from_str::<RecordingLine>(&line)? {
            RecordingLine::Manifest { manifest } => manifest,
            RecordingLine::Generation { .. } => return Err(RecordingError::MissingManifest),
        };
        manifest.validate()?;
        Ok(Self {
            reader,
            manifest,
            line_number,
            next_sequence: 0,
            last_generation: None,
            finished: false,
        })
    }

    /// Return the immutable recording manifest.
    pub fn manifest(&self) -> &RecordingManifest {
        &self.manifest
    }

    /// Read the next generation, or `None` at end of stream.
    pub fn read_generation(&mut self) -> Result<Option<GenerationRecord>, RecordingError> {
        if self.finished {
            return Ok(None);
        }
        let Some(line) = read_nonempty_line(&mut self.reader, &mut self.line_number)? else {
            self.finished = true;
            return Ok(None);
        };
        let record = match serde_json::from_str::<RecordingLine>(&line)? {
            RecordingLine::Manifest { .. } => return Err(RecordingError::DuplicateManifest),
            RecordingLine::Generation { record } => *record,
        };
        record.validate()?;
        if record.circuit_sequence != self.next_sequence {
            return Err(RecordingError::SequenceMismatch {
                expected: self.next_sequence,
                actual: record.circuit_sequence,
            });
        }
        if let Some(previous) = self.last_generation {
            if record.update.generation <= previous {
                return Err(RecordingError::NonIncreasingGeneration {
                    previous,
                    actual: record.update.generation,
                });
            }
        }
        self.next_sequence += 1;
        self.last_generation = Some(record.update.generation);
        Ok(Some(record))
    }
}

impl RecordingReader<BufReader<File>> {
    /// Open a JSON Lines recording file and read its manifest.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, RecordingError> {
        Self::new(BufReader::new(File::open(path)?))
    }
}

impl<R: BufRead> Iterator for RecordingReader<R> {
    type Item = Result<GenerationRecord, RecordingError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.read_generation() {
            Ok(Some(record)) => Some(Ok(record)),
            Ok(None) => None,
            Err(error) => {
                self.finished = true;
                Some(Err(error))
            }
        }
    }
}

fn write_line<W: Write, T: Serialize>(writer: &mut W, line: &T) -> Result<(), RecordingError> {
    serde_json::to_writer(&mut *writer, line)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn read_nonempty_line<R: BufRead>(
    reader: &mut R,
    line_number: &mut usize,
) -> Result<Option<String>, RecordingError> {
    loop {
        let mut line = String::new();
        let read = reader.read_line(&mut line)?;
        if read == 0 {
            return Ok(None);
        }
        *line_number += 1;
        if !line.trim().is_empty() {
            return Ok(Some(line));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufReader, Cursor};

    use hypercube::{ExecutionMode, InputRow, NodeSpec, Transform};

    use crate::{SnapshotDigest, TriggerFrame};

    use super::*;

    fn manifest() -> RecordingManifest {
        RecordingManifest::new(
            "run-1",
            "build-1",
            "config-1",
            "layout-1",
            vec![ThresholdTriggerSpec::new("high", "score", 1.0, 0.5, 2).unwrap()],
        )
        .unwrap()
        .with_metadata("aeron_recording_id", "17")
        .unwrap()
    }

    fn record(sequence: u64, generation: u64) -> GenerationRecord {
        let mut source_positions = BTreeMap::new();
        source_positions.insert("market-data".to_owned(), generation as i64 * 1_024);
        GenerationRecord {
            circuit_sequence: sequence,
            update: Update {
                generation,
                observed_at_ms: generation as i64 * 100,
                mode: ExecutionMode::Live,
                rows: vec![InputRow::new("A", generation as i64 * 100).with_field("score", 1.0)],
                nodes: vec![NodeSpec::field("score", "score", Transform::Identity)],
            },
            source_positions,
            expected: TriggerFrame {
                generation,
                snapshot: SnapshotDigest {
                    algorithm: "test".to_owned(),
                    hash: generation,
                    values: 1,
                    missing: 0,
                },
                transitions: Vec::new(),
                states: Vec::new(),
            },
        }
    }

    #[test]
    fn recording_round_trips_as_a_stream() {
        let mut bytes = Vec::new();
        {
            let mut writer = RecordingWriter::new(&mut bytes, manifest()).unwrap();
            writer.write_generation(&record(0, 1)).unwrap();
            writer.write_generation(&record(1, 2)).unwrap();
            writer.flush().unwrap();
        }

        let cursor = Cursor::new(bytes);
        let mut reader = RecordingReader::new(BufReader::new(cursor)).unwrap();
        assert_eq!(reader.manifest(), &manifest());
        assert_eq!(reader.read_generation().unwrap(), Some(record(0, 1)));
        assert_eq!(reader.read_generation().unwrap(), Some(record(1, 2)));
        assert_eq!(reader.read_generation().unwrap(), None);
    }

    #[test]
    fn writer_rejects_sequence_gaps() {
        let mut bytes = Vec::new();
        let mut writer = RecordingWriter::new(&mut bytes, manifest()).unwrap();
        let error = writer.write_generation(&record(1, 1)).unwrap_err();
        assert!(matches!(
            error,
            RecordingError::SequenceMismatch {
                expected: 0,
                actual: 1
            }
        ));
    }

    #[test]
    fn writer_rejects_nonfinite_values_before_writing_a_generation() {
        let manifest_len = RecordingWriter::new(Vec::new(), manifest())
            .unwrap()
            .into_inner()
            .unwrap()
            .len();
        let mut writer = RecordingWriter::new(Vec::new(), manifest()).unwrap();
        let mut invalid = record(0, 1);
        invalid.update.rows[0]
            .fields
            .insert("score".to_owned(), f64::NAN);

        let error = writer.write_generation(&invalid).unwrap_err();
        assert!(matches!(error, RecordingError::NonFiniteValue { .. }));
        assert_eq!(writer.into_inner().unwrap().len(), manifest_len);
    }
}
