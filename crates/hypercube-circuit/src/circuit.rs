use std::collections::VecDeque;
use std::hint;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use disruptor::wait_strategies::WaitStrategy;
use disruptor::{build_single_producer, Producer, Sequence, SingleConsumerBarrier, SingleProducer};
use hypercube::{ExecutionMode, Snapshot};
use thiserror::Error;

/// Per-generation context supplied to a stateful frame processor.
///
/// Stateful code should use `observed_at_ms` as its logical clock instead of
/// consulting wall time. Replay changes `mode` but preserves the original
/// generation and observation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameContext {
    /// Disruptor sequence assigned to this frame.
    pub circuit_sequence: i64,
    /// Hypercube generation carried by the snapshot.
    pub generation: u64,
    /// Original logical observation time.
    pub observed_at_ms: i64,
    /// Live, replay, or batch execution context.
    pub mode: ExecutionMode,
}

/// Stateful callback invoked once for every coherent Hypercube snapshot.
///
/// Correctness-lane implementations should be deterministic and must not
/// perform irreversible external effects. Publish their output through a
/// separately guarded adapter instead.
pub trait FrameProcessor: Send + 'static {
    /// Deterministic result returned for one processed frame.
    type Output: Send + 'static;

    /// Process `snapshot` using the supplied logical context.
    fn process(&mut self, context: FrameContext, snapshot: &Snapshot) -> Self::Output;
}

/// Consumer behavior while the circuit is waiting for its next generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitWaitStrategy {
    /// Poll continuously for the lowest latency, occupying a processor core.
    BusySpin,
    /// Poll with a CPU spin-loop hint, trading some latency for friendlier
    /// hyper-thread and power behavior.
    SpinLoop,
    /// Sleep between polls for lower idle CPU use and higher wake-up latency.
    Sleep(Duration),
}

impl WaitStrategy for CircuitWaitStrategy {
    fn wait_for(&self, _sequence: Sequence) {
        match self {
            Self::BusySpin => {}
            Self::SpinLoop => hint::spin_loop(),
            Self::Sleep(duration) => thread::sleep(*duration),
        }
    }
}

/// Runtime settings for [`DisruptorCircuit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CircuitConfig {
    /// Number of preallocated ring slots; must be a power of two of at least 2.
    pub ring_size: usize,
    /// Consumer behavior when no frame is available.
    pub wait_strategy: CircuitWaitStrategy,
    /// Maximum wait for a submitted generation to complete.
    pub completion_timeout: Duration,
}

impl Default for CircuitConfig {
    fn default() -> Self {
        Self {
            ring_size: 16,
            wait_strategy: CircuitWaitStrategy::Sleep(Duration::from_micros(50)),
            completion_timeout: Duration::from_secs(5),
        }
    }
}

/// One completed circuit callback and its ordering coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessedFrame<O> {
    /// Disruptor sequence assigned at submission.
    pub circuit_sequence: i64,
    /// Hypercube generation processed by the callback.
    pub generation: u64,
    /// Processor-specific deterministic output.
    pub output: O,
}

/// Configuration, backpressure, or completion error from a circuit.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CircuitError {
    /// Ring size did not meet the Disruptor layout requirement.
    #[error("ring size {0} must be a power of two and at least 2")]
    InvalidRingSize(usize),
    /// A nonblocking submission found no free ring slot.
    #[error("circuit ring is full while submitting generation {generation}")]
    RingFull {
        /// Generation that could not be submitted.
        generation: u64,
    },
    /// No submitted generation is waiting for completion.
    #[error("no circuit generation is pending")]
    NothingPending,
    /// The processor thread stopped without returning its result.
    #[error("circuit processor disconnected")]
    ProcessorDisconnected,
    /// The submitted generation exceeded its completion deadline.
    #[error("generation {generation} did not complete before the circuit deadline")]
    CompletionTimeout {
        /// Generation that timed out.
        generation: u64,
    },
    /// Completion order differed from submission order.
    #[error(
        "circuit completion mismatch: expected sequence {expected_sequence} generation \
         {expected_generation}, got sequence {actual_sequence} generation {actual_generation}"
    )]
    CompletionMismatch {
        /// Expected Disruptor sequence.
        expected_sequence: i64,
        /// Expected Hypercube generation.
        expected_generation: u64,
        /// Returned Disruptor sequence.
        actual_sequence: i64,
        /// Returned Hypercube generation.
        actual_generation: u64,
    },
}

#[derive(Default)]
struct FrameEvent {
    snapshot: Option<Arc<Snapshot>>,
}

/// Single-producer Disruptor runtime for coherent Hypercube generations.
///
/// Submission is nonblocking and returns [`CircuitError::RingFull`] rather
/// than silently waiting behind stale work. [`Self::process`] provides the
/// lockstep submit-and-wait path used by deterministic replay.
pub struct DisruptorCircuit<O: Send + 'static> {
    producer: Option<SingleProducer<FrameEvent, SingleConsumerBarrier>>,
    completed: Receiver<ProcessedFrame<O>>,
    pending: VecDeque<(i64, u64)>,
    completion_timeout: Duration,
}

impl<O: Send + 'static> DisruptorCircuit<O> {
    /// Build a circuit around one stateful processor.
    pub fn with_processor<P>(processor: P, config: CircuitConfig) -> Result<Self, CircuitError>
    where
        P: FrameProcessor<Output = O>,
    {
        if config.ring_size < 2 || !config.ring_size.is_power_of_two() {
            return Err(CircuitError::InvalidRingSize(config.ring_size));
        }

        let (completed_tx, completed) = mpsc::channel();
        let handler = move |processor: &mut P,
                            event: &FrameEvent,
                            circuit_sequence: i64,
                            _end_of_batch: bool| {
            let Some(snapshot) = event.snapshot.as_deref() else {
                return;
            };
            let context = FrameContext {
                circuit_sequence,
                generation: snapshot.generation,
                observed_at_ms: snapshot.observed_at_ms,
                mode: snapshot.mode,
            };
            let output = processor.process(context, snapshot);
            let _ = completed_tx.send(ProcessedFrame {
                circuit_sequence,
                generation: snapshot.generation,
                output,
            });
        };
        let producer =
            build_single_producer(config.ring_size, FrameEvent::default, config.wait_strategy)
                .handle_events_and_state_with(handler, move || processor)
                .build();

        Ok(Self {
            producer: Some(producer),
            completed,
            pending: VecDeque::new(),
            completion_timeout: config.completion_timeout,
        })
    }

    /// Submit one immutable snapshot without waiting for processing.
    ///
    /// The returned sequence and generation can be matched to
    /// [`Self::receive`] output.
    pub fn try_submit(&mut self, snapshot: Arc<Snapshot>) -> Result<i64, CircuitError> {
        let generation = snapshot.generation;
        let sequence = self
            .producer
            .as_mut()
            .expect("producer remains present until circuit drop")
            .try_publish(move |event| event.snapshot = Some(snapshot))
            .map_err(|_| CircuitError::RingFull { generation })?;
        self.pending.push_back((sequence, generation));
        Ok(sequence)
    }

    /// Wait for the oldest submitted generation to complete.
    pub fn receive(&mut self) -> Result<ProcessedFrame<O>, CircuitError> {
        let (expected_sequence, expected_generation) = self
            .pending
            .front()
            .copied()
            .ok_or(CircuitError::NothingPending)?;
        let completed = match self.completed.recv_timeout(self.completion_timeout) {
            Ok(completed) => completed,
            Err(RecvTimeoutError::Timeout) => {
                return Err(CircuitError::CompletionTimeout {
                    generation: expected_generation,
                });
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(CircuitError::ProcessorDisconnected);
            }
        };
        if completed.circuit_sequence != expected_sequence
            || completed.generation != expected_generation
        {
            return Err(CircuitError::CompletionMismatch {
                expected_sequence,
                expected_generation,
                actual_sequence: completed.circuit_sequence,
                actual_generation: completed.generation,
            });
        }
        self.pending.pop_front();
        Ok(completed)
    }

    /// Submit one snapshot and wait for its stateful processing to complete.
    pub fn process(&mut self, snapshot: Arc<Snapshot>) -> Result<ProcessedFrame<O>, CircuitError> {
        self.try_submit(snapshot)?;
        self.receive()
    }

    /// Return the number of submitted generations awaiting completion.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }
}

impl<O: Send + 'static> Drop for DisruptorCircuit<O> {
    fn drop(&mut self) {
        // Drop joins the managed consumer while the completion receiver is
        // still alive, so a final callback can finish cleanly.
        drop(self.producer.take());
    }
}

#[cfg(test)]
mod tests {
    use hypercube::{ExecutionMode, Snapshot};

    use super::*;

    struct GenerationProcessor;

    impl FrameProcessor for GenerationProcessor {
        type Output = (u64, i64);

        fn process(&mut self, context: FrameContext, _snapshot: &Snapshot) -> Self::Output {
            (context.generation, context.observed_at_ms)
        }
    }

    fn snapshot(generation: u64) -> Arc<Snapshot> {
        Arc::new(Snapshot {
            generation,
            observed_at_ms: generation as i64 * 100,
            mode: ExecutionMode::Replay,
            entity_count: 0,
            values: Vec::new(),
            statuses: Vec::new(),
        })
    }

    #[test]
    fn rejects_invalid_ring_size() {
        let result = DisruptorCircuit::with_processor(
            GenerationProcessor,
            CircuitConfig {
                ring_size: 3,
                ..CircuitConfig::default()
            },
        );
        assert!(matches!(result, Err(CircuitError::InvalidRingSize(3))));
    }

    #[test]
    fn processes_frames_in_submission_order() {
        let mut circuit =
            DisruptorCircuit::with_processor(GenerationProcessor, CircuitConfig::default())
                .unwrap();
        circuit.try_submit(snapshot(1)).unwrap();
        circuit.try_submit(snapshot(2)).unwrap();

        let first = circuit.receive().unwrap();
        let second = circuit.receive().unwrap();
        assert_eq!(first.output, (1, 100));
        assert_eq!(second.output, (2, 200));
        assert_eq!(first.circuit_sequence, 0);
        assert_eq!(second.circuit_sequence, 1);
    }

    struct BlockingProcessor {
        started: mpsc::Sender<u64>,
        release: mpsc::Receiver<()>,
    }

    impl FrameProcessor for BlockingProcessor {
        type Output = u64;

        fn process(&mut self, context: FrameContext, _snapshot: &Snapshot) -> Self::Output {
            self.started.send(context.generation).unwrap();
            self.release
                .recv_timeout(Duration::from_secs(2))
                .expect("test releases each blocked generation");
            context.generation
        }
    }

    #[test]
    fn full_ring_is_reported_without_overwriting_work() {
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let processor = BlockingProcessor {
            started: started_tx,
            release: release_rx,
        };
        let mut circuit = DisruptorCircuit::with_processor(
            processor,
            CircuitConfig {
                ring_size: 2,
                ..CircuitConfig::default()
            },
        )
        .unwrap();

        circuit.try_submit(snapshot(1)).unwrap();
        assert_eq!(started_rx.recv_timeout(Duration::from_secs(1)).unwrap(), 1);
        circuit.try_submit(snapshot(2)).unwrap();
        assert!(matches!(
            circuit.try_submit(snapshot(3)),
            Err(CircuitError::RingFull { generation: 3 })
        ));

        release_tx.send(()).unwrap();
        assert_eq!(circuit.receive().unwrap().output, 1);
        assert_eq!(started_rx.recv_timeout(Duration::from_secs(1)).unwrap(), 2);
        release_tx.send(()).unwrap();
        assert_eq!(circuit.receive().unwrap().output, 2);
    }
}
