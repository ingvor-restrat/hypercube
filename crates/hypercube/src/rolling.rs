//! Fixed-capacity online statistics for stateful streaming adapters.
//!
//! [`RollingMoments`] is not an implicit stateful graph node. It is a reusable
//! primitive for an owner that already has an explicit state boundary, such as
//! a circuit callback, simulator, or bounded backtest.

use thiserror::Error;

/// Invalid input or arithmetic failure from [`RollingMoments`].
#[derive(Debug, Clone, Error, PartialEq)]
pub enum RollingError {
    /// A rolling window must retain at least one observation.
    #[error("rolling window must be positive")]
    EmptyWindow,
    /// Only finite observations can enter or be scored against the window.
    #[error("rolling observation must be finite, got {0}")]
    NonFinite(f64),
    /// The supplied values exceeded finite `f64` moment arithmetic.
    #[error("rolling moment arithmetic overflowed")]
    NumericalOverflow,
}

/// Fixed-capacity rolling mean and variance with constant-time replacement.
///
/// The implementation keeps a preallocated circular buffer and the corrected
/// sum of squares `M2`. Insertions and evictions use the centered updating
/// identities associated with Welford and Chan--Golub--LeVeque rather than the
/// cancellation-prone `sum(x²) - sum(x)² / n` formula. A bounded periodic
/// rebuild limits accumulated roundoff, while an internal retained origin
/// keeps small variations precise around a large common level. Updates remain
/// amortized constant time.
#[derive(Debug, Clone)]
pub struct RollingMoments {
    values: Vec<f64>,
    len: usize,
    next: usize,
    origin: f64,
    mean_offset: f64,
    m2: f64,
    replacements_since_rebuild: usize,
}

impl RollingMoments {
    /// Allocate an empty rolling window with fixed `capacity`.
    pub fn new(capacity: usize) -> Result<Self, RollingError> {
        if capacity == 0 {
            return Err(RollingError::EmptyWindow);
        }
        Ok(Self {
            values: vec![0.0; capacity],
            len: 0,
            next: 0,
            origin: 0.0,
            mean_offset: 0.0,
            m2: 0.0,
            replacements_since_rebuild: 0,
        })
    }

    /// Return the fixed number of observations retained by a full window.
    pub fn capacity(&self) -> usize {
        self.values.len()
    }

    /// Return the number of observations currently retained.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return whether the window contains no observations.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return whether the next insertion will evict the oldest observation.
    pub fn is_full(&self) -> bool {
        self.len == self.capacity()
    }

    /// Return the current arithmetic mean, or `None` for an empty window.
    pub fn mean(&self) -> Option<f64> {
        (self.len > 0).then_some(self.origin + self.mean_offset)
    }

    /// Return the population variance, or `None` for an empty window.
    pub fn population_variance(&self) -> Option<f64> {
        (self.len > 0).then_some((self.m2 / self.len as f64).max(0.0))
    }

    /// Return the sample variance, or `None` until two values are present.
    pub fn sample_variance(&self) -> Option<f64> {
        (self.len > 1).then_some((self.m2 / (self.len - 1) as f64).max(0.0))
    }

    /// Return the population standard deviation, or `None` when it is zero or
    /// fewer than two observations are present.
    pub fn population_standard_deviation(&self) -> Option<f64> {
        self.population_variance()
            .filter(|_| self.len > 1)
            .map(f64::sqrt)
            .filter(|deviation| *deviation > 0.0)
    }

    /// Standardize `value` against the observations already in the window.
    ///
    /// Calling this before [`Self::push`] gives a no-look-ahead score for a new
    /// observation. `None` means the existing window has insufficient or zero
    /// variance.
    pub fn z_score(&self, value: f64) -> Result<Option<f64>, RollingError> {
        validate_value(value)?;
        let Some(deviation) = self.population_standard_deviation() else {
            return Ok(None);
        };
        let centered = (value - self.origin) - self.mean_offset;
        let score = centered / deviation;
        if !score.is_finite() {
            return Err(RollingError::NumericalOverflow);
        }
        Ok(Some(score))
    }

    /// Insert `value`, returning the evicted observation when the window was
    /// already full.
    ///
    /// Invalid or overflowing input leaves the window unchanged.
    pub fn push(&mut self, value: f64) -> Result<Option<f64>, RollingError> {
        validate_value(value)?;
        let origin = if self.is_empty() { value } else { self.origin };
        let value_offset = value - origin;
        if !value_offset.is_finite() {
            return Err(RollingError::NumericalOverflow);
        }
        let evicted = self.is_full().then_some(self.values[self.next]);
        let (base_len, base_mean, base_m2) = match evicted {
            Some(oldest) => {
                let oldest_offset = oldest - origin;
                remove_moment(self.len, self.mean_offset, self.m2, oldest_offset)
            }
            None => (self.len, self.mean_offset, self.m2),
        };
        let (new_len, new_mean_offset, new_m2) =
            add_moment(base_len, base_mean, base_m2, value_offset);
        if !new_mean_offset.is_finite()
            || !(origin + new_mean_offset).is_finite()
            || !new_m2.is_finite()
        {
            return Err(RollingError::NumericalOverflow);
        }

        self.values[self.next] = value;
        self.next = (self.next + 1) % self.capacity();
        self.len = new_len;
        self.origin = origin;
        self.mean_offset = new_mean_offset;
        self.m2 = new_m2.max(0.0);
        if evicted.is_some() {
            self.replacements_since_rebuild = self.replacements_since_rebuild.saturating_add(1);
            let rebuild_interval = self.capacity().saturating_mul(1_024).max(1_024);
            if self.replacements_since_rebuild >= rebuild_interval {
                self.rebuild();
            }
        }
        Ok(evicted)
    }

    fn rebuild(&mut self) {
        let values = &self.values[..self.len];
        if let Some((origin, mean_offset, m2)) = corrected_two_pass(values) {
            self.origin = origin;
            self.mean_offset = mean_offset;
            self.m2 = m2;
            self.replacements_since_rebuild = 0;
        }
    }
}

fn validate_value(value: f64) -> Result<(), RollingError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(RollingError::NonFinite(value))
    }
}

fn add_moment(len: usize, mean: f64, m2: f64, value: f64) -> (usize, f64, f64) {
    let new_len = len + 1;
    let delta = value - mean;
    let new_mean = mean + delta / new_len as f64;
    let new_m2 = m2 + delta * (value - new_mean);
    (new_len, new_mean, new_m2)
}

fn remove_moment(len: usize, mean: f64, m2: f64, value: f64) -> (usize, f64, f64) {
    debug_assert!(len > 0);
    if len == 1 {
        return (0, 0.0, 0.0);
    }
    let new_len = len - 1;
    // This rearrangement avoids multiplying a potentially large mean by the
    // window length before subtracting the outgoing observation.
    let new_mean = mean + (mean - value) / new_len as f64;
    let new_m2 = m2 - (value - mean) * (value - new_mean);
    (new_len, new_mean, new_m2)
}

fn corrected_two_pass(values: &[f64]) -> Option<(f64, f64, f64)> {
    let (&origin, rest) = values.split_first()?;
    let shifted_sum = rest.iter().map(|value| value - origin).sum::<f64>();
    let mean_offset = shifted_sum / values.len() as f64;
    let mut squared_deviations = 0.0;
    let mut deviation_sum = 0.0;
    for value in values {
        let deviation = (value - origin) - mean_offset;
        squared_deviations += deviation * deviation;
        deviation_sum += deviation;
    }
    let correction = deviation_sum * deviation_sum / values.len() as f64;
    let m2 = (squared_deviations - correction).max(0.0);
    let mean = origin + mean_offset;
    (mean.is_finite() && mean_offset.is_finite() && m2.is_finite()).then_some((
        origin,
        mean_offset,
        m2,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected_moments(values: &[f64]) -> (f64, f64) {
        let (origin, mean_offset, m2) = corrected_two_pass(values).unwrap();
        (origin + mean_offset, m2 / values.len() as f64)
    }

    #[test]
    fn rolling_moments_match_a_two_pass_window() {
        for capacity in [2, 8, 63] {
            let mut rolling = RollingMoments::new(capacity).unwrap();
            let mut expected = Vec::new();
            for index in 0..20_000 {
                let value = (index as f64 * 0.037).sin() + (index % 17) as f64 * 1e-4;
                rolling.push(value).unwrap();
                expected.push(value);
                if expected.len() > capacity {
                    expected.remove(0);
                }
                let (mean, variance) = expected_moments(&expected);
                assert!((rolling.mean().unwrap() - mean).abs() < 1e-11);
                assert!((rolling.population_variance().unwrap() - variance).abs() < 1e-11);
            }
        }
    }

    #[test]
    fn large_offsets_retain_small_window_variance() {
        let capacity = 8;
        let mut rolling = RollingMoments::new(capacity).unwrap();
        let mut expected = Vec::new();
        for index in 0..20_000 {
            let value = 1e12 + ((index % 17) as f64 - 8.0) * 0.001;
            rolling.push(value).unwrap();
            expected.push(value);
            if expected.len() > capacity {
                expected.remove(0);
            }
            let (mean, variance) = expected_moments(&expected);
            assert!((rolling.mean().unwrap() - mean).abs() <= 0.001);
            let observed_variance = rolling.population_variance().unwrap();
            assert!(
                (observed_variance - variance).abs() < 2e-6,
                "index={index} observed={observed_variance:e} expected={variance:e}"
            );
        }
    }

    #[test]
    fn scores_before_insertion_without_look_ahead() {
        let mut rolling = RollingMoments::new(3).unwrap();
        rolling.push(1.0).unwrap();
        rolling.push(2.0).unwrap();
        rolling.push(3.0).unwrap();
        assert_eq!(rolling.z_score(4.0).unwrap(), Some(2.449489742783178));
        assert_eq!(rolling.push(4.0).unwrap(), Some(1.0));
        assert!((rolling.mean().unwrap() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn rejects_invalid_input_without_mutating_state() {
        assert_eq!(
            RollingMoments::new(0).unwrap_err(),
            RollingError::EmptyWindow
        );
        let mut rolling = RollingMoments::new(4).unwrap();
        rolling.push(2.0).unwrap();
        assert!(matches!(
            rolling.push(f64::NAN),
            Err(RollingError::NonFinite(value)) if value.is_nan()
        ));
        assert_eq!(rolling.len(), 1);
        assert_eq!(rolling.mean(), Some(2.0));
    }

    #[test]
    fn constant_windows_have_no_defined_z_score() {
        let mut rolling = RollingMoments::new(3).unwrap();
        for _ in 0..3 {
            rolling.push(5.0).unwrap();
        }
        assert_eq!(rolling.population_variance(), Some(0.0));
        assert_eq!(rolling.z_score(6.0).unwrap(), None);
    }

    #[test]
    fn representable_sub_epsilon_scales_can_be_scored() {
        let mut rolling = RollingMoments::new(2).unwrap();
        rolling.push(0.0).unwrap();
        rolling.push(1e-20).unwrap();
        assert_eq!(rolling.z_score(2e-20).unwrap(), Some(3.0));
    }
}
