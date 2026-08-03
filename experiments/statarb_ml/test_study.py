import unittest

import numpy as np

from robustness import derived_seeds
from study import (
    StudyConfig,
    downside_adjusted_quantile_score,
    expected_barrier_score,
    generate_scenario,
    make_events,
    make_folds,
    split_events,
    triple_barrier_outcome,
)


class StudyTests(unittest.TestCase):
    def test_robustness_seeds_are_deterministic_and_distinct(self):
        first = derived_seeds(5, master_seed=17)
        self.assertEqual(first, derived_seeds(5, master_seed=17))
        self.assertEqual(len(first), len(set(first)))

    def test_z_score_uses_only_the_preceding_window(self):
        config = StudyConfig.quick(seed=7)
        original = generate_scenario("stationary_ou", config)
        events = make_events(original, config)
        target = events.iloc[len(events) // 2]
        pair, time = int(target.pair), int(target.time)
        prior = original.residuals[pair, time - config.z_window : time]
        expected = (original.residuals[pair, time] - prior.mean()) / prior.std(ddof=0)
        self.assertAlmostEqual(target.z, expected)

    def test_features_ignore_future_observations(self):
        config = StudyConfig.quick(seed=7)
        original = generate_scenario("stationary_ou", config)
        events = make_events(original, config)
        target = events.iloc[len(events) // 2]
        changed = generate_scenario("stationary_ou", config)
        changed.residuals[int(target.pair), int(target.time) + 1 :] += 10_000.0
        changed_events = make_events(changed, config)
        matched = changed_events.loc[
            (changed_events.pair == target.pair) & (changed_events.time == target.time)
        ].iloc[0]
        for feature in ["z", "abs_z", "delta_1", "phi_hat", "market_vol"]:
            self.assertEqual(target[feature], matched[feature])

    def test_triple_barrier_uses_first_touch(self):
        path = np.asarray([0.0, 0.4, 0.8, -2.0])
        label, net_return, duration, exit_time = triple_barrier_outcome(
            path,
            start=0,
            side=1.0,
            scale=1.0,
            horizon=3,
            profit_barrier=0.75,
            stop_barrier=1.0,
            round_trip_cost=0.1,
        )
        self.assertEqual((label, duration, exit_time), (1, 2, 2))
        self.assertAlmostEqual(net_return, 0.7)

    def test_quantile_score_penalizes_lower_tail_uncertainty(self):
        compact = downside_adjusted_quantile_score(np.asarray([[-0.2, 0.5, 0.8]]))[0]
        wide = downside_adjusted_quantile_score(np.asarray([[-1.0, 0.5, 1.1]]))[0]
        self.assertGreater(compact, wide)

    def test_barrier_score_is_probability_weighted_net_payoff(self):
        probability = np.asarray([[0.2, 0.3, 0.5]])
        score = expected_barrier_score(
            probability,
            classes=[0, 1, 2],
            class_payoffs={0: -1.1, 1: -0.1, 2: 0.7},
        )[0]
        self.assertAlmostEqual(score, 0.10)

    def test_temporal_blocks_are_purged_by_a_full_label_horizon(self):
        config = StudyConfig.quick(seed=11)
        events = make_events(generate_scenario("stationary_ou", config), config)
        fold = make_folds(config)[0]
        split = split_events(events, fold)
        self.assertLess(split["train"].time.max() + config.horizon, split["validation"].time.min())
        self.assertLess(
            split["validation"].time.max() + config.horizon,
            split["calibration"].time.min(),
        )
        self.assertLess(
            split["calibration"].time.max() + config.horizon,
            split["selection"].time.min(),
        )
        self.assertLess(
            split["selection"].time.max() + config.horizon,
            split["test"].time.min(),
        )


if __name__ == "__main__":
    unittest.main()
