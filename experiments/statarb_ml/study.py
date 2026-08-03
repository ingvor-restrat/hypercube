"""Leakage-controlled statistical-arbitrage model comparison.

This module deliberately keeps research dependencies out of the Rust runtime.
It generates controlled residual processes, creates causal features and
path-dependent labels, trains tabular models on expanding windows, calibrates
them on later data, selects policies on a later block, and evaluates only on
untouched future blocks.
"""

from __future__ import annotations

import json
import math
import platform
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import catboost
import lightgbm
import matplotlib
import numpy as np
import pandas as pd
import sklearn

matplotlib.use("Agg")

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, mean_absolute_error, mean_pinball_loss, mean_squared_error

FEATURE_COLUMNS = [
    "z",
    "abs_z",
    "delta_1",
    "delta_3",
    "delta_10",
    "vol_ratio_5_20",
    "phi_hat",
    "half_life_hat",
    "market_vol",
]
QUANTILES = np.asarray([0.2, 0.5, 0.8], dtype=float)
QUANTILE_DOWNSIDE_PENALTY = 0.25
LANE_ORDER = [
    "rolling_z",
    "lightgbm_point",
    "catboost_point",
    "lightgbm_huber",
    "catboost_huber",
    "lightgbm_quantile",
    "catboost_quantile",
    "lightgbm_barrier",
    "catboost_barrier",
]


@dataclass(frozen=True)
class StudyConfig:
    seed: int = 335_341
    pairs: int = 20
    observations: int = 3_300
    z_window: int = 20
    feature_window: int = 60
    horizon: int = 12
    candidate_z: float = 1.0
    profit_barrier: float = 0.75
    stop_barrier: float = 1.0
    round_trip_cost: float = 0.08
    iterations: int = 350
    min_selection_trades: int = 30

    @classmethod
    def quick(cls, seed: int = 335_341) -> "StudyConfig":
        return cls(
            seed=seed,
            pairs=8,
            observations=1_700,
            iterations=60,
            min_selection_trades=10,
        )


@dataclass(frozen=True)
class Fold:
    fold: int
    train_end: int
    validation_start: int
    validation_end: int
    calibration_start: int
    calibration_end: int
    selection_start: int
    selection_end: int
    test_start: int
    test_end: int


@dataclass
class ScenarioData:
    name: str
    residuals: np.ndarray
    market_vol: np.ndarray
    latent_regime: np.ndarray


def generate_scenario(name: str, config: StudyConfig) -> ScenarioData:
    """Generate a stationary negative control or observable regime stress."""

    if name not in {"stationary_ou", "regime_switching"}:
        raise ValueError(f"unknown scenario {name}")
    rng = np.random.default_rng(config.seed + (0 if name == "stationary_ou" else 10_000))
    count = config.observations
    pair_count = config.pairs

    latent_regime = np.zeros(count, dtype=np.int8)
    if name == "regime_switching":
        transition = np.asarray(
            [
                [0.965, 0.030, 0.005],
                [0.060, 0.890, 0.050],
                [0.150, 0.100, 0.750],
            ]
        )
        for time in range(1, count):
            latent_regime[time] = rng.choice(3, p=transition[latent_regime[time - 1]])

    common_innovation = rng.normal(size=count)
    market_vol = np.zeros(count, dtype=float)
    common_state = 0.0
    regime_volatility = np.asarray([1.0, 1.85, 2.60])
    for time in range(1, count):
        multiplier = regime_volatility[latent_regime[time]]
        common_state = 0.82 * common_state + multiplier * common_innovation[time]
        observed_proxy = (
            0.65 * abs(common_state)
            + 0.35 * multiplier
            + 0.20 * abs(rng.normal())
        )
        market_vol[time] = 0.94 * market_vol[time - 1] + 0.06 * observed_proxy

    residuals = np.zeros((pair_count, count), dtype=float)
    loadings = rng.uniform(-0.35, 0.35, size=pair_count)
    for pair in range(pair_count):
        if name == "stationary_ou":
            base_phi = 0.90
            base_sigma = 0.0040
        else:
            base_phi = 0.84 + 0.02 * (pair % 5)
            base_sigma = 0.0032 + 0.00035 * (pair % 4)
        for time in range(1, count):
            state = int(latent_regime[time])
            if name == "stationary_ou":
                phi = base_phi
                sigma_multiplier = 1.0
            elif state == 0:
                phi = base_phi
                sigma_multiplier = 1.0
            elif state == 1:
                phi = min(0.985, base_phi + 0.10)
                sigma_multiplier = 1.8
            else:
                # A short positive-control breakdown: residuals can trend
                # briefly, but the state is not handed directly to the model.
                phi = 1.025
                sigma_multiplier = 2.5
            if name == "regime_switching" and state > 0:
                # Unit-variance Student-t shocks make the stress case
                # genuinely heavy-tailed without changing its scale by fiat.
                idiosyncratic = rng.standard_t(df=4.0) / math.sqrt(2.0)
            else:
                idiosyncratic = rng.normal()
            innovation = idiosyncratic + loadings[pair] * common_innovation[time]
            residuals[pair, time] = (
                phi * residuals[pair, time - 1]
                + base_sigma * sigma_multiplier * innovation
            )

    return ScenarioData(name, residuals, market_vol, latent_regime)


def _population_std(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values - values.mean()))))


def _estimate_phi(values: np.ndarray) -> float:
    left = values[:-1]
    right = values[1:]
    denominator = float(np.sum(np.square(left - left.mean())))
    if denominator <= np.finfo(float).eps:
        return 0.0
    numerator = float(np.sum((left - left.mean()) * (right - right.mean())))
    return float(np.clip(numerator / denominator, -0.999, 1.05))


def _half_life(phi: float) -> float:
    if not 0.0 < phi < 0.999:
        return 100.0
    return float(np.clip(math.log(0.5) / math.log(phi), 0.0, 100.0))


def triple_barrier_outcome(
    path: np.ndarray,
    start: int,
    side: float,
    scale: float,
    horizon: int,
    profit_barrier: float,
    stop_barrier: float,
    round_trip_cost: float,
) -> tuple[int, float, int, int]:
    """Return label, normalized net return, duration, and exit index."""

    initial = float(path[start])
    final_step = min(start + horizon, len(path) - 1)
    label = 0
    exit_index = final_step
    gross = side * (float(path[final_step]) - initial) / scale
    for index in range(start + 1, final_step + 1):
        candidate = side * (float(path[index]) - initial) / scale
        if candidate >= profit_barrier:
            label = 1
            gross = candidate
            exit_index = index
            break
        if candidate <= -stop_barrier:
            label = -1
            gross = candidate
            exit_index = index
            break
    return label, gross - round_trip_cost, exit_index - start, exit_index


def make_events(data: ScenarioData, config: StudyConfig) -> pd.DataFrame:
    """Create causal candidate features and future event outcomes."""

    records: list[dict[str, float | int | str]] = []
    warmup = max(config.feature_window + 1, config.z_window + 1)
    for pair, path in enumerate(data.residuals):
        for time in range(warmup, config.observations - config.horizon):
            prior = path[time - config.z_window : time]
            scale = _population_std(prior)
            if not np.isfinite(scale) or scale <= np.finfo(float).eps:
                continue
            current = float(path[time])
            z = (current - float(prior.mean())) / scale
            if abs(z) < config.candidate_z:
                continue

            history = path[time - config.feature_window : time]
            differences = np.diff(path[time - 20 : time + 1])
            vol_20 = _population_std(differences)
            vol_5 = _population_std(differences[-5:])
            phi_hat = _estimate_phi(history)
            side = -1.0 if z > 0.0 else 1.0
            label, net_return, duration, exit_time = triple_barrier_outcome(
                path,
                time,
                side,
                scale,
                config.horizon,
                config.profit_barrier,
                config.stop_barrier,
                config.round_trip_cost,
            )
            records.append(
                {
                    "scenario": data.name,
                    "pair": pair,
                    "time": time,
                    "exit_time": exit_time,
                    "duration": duration,
                    "label": label,
                    "net_return": net_return,
                    "z": z,
                    "abs_z": abs(z),
                    "delta_1": (current - float(path[time - 1])) / scale,
                    "delta_3": (current - float(path[time - 3])) / scale,
                    "delta_10": (current - float(path[time - 10])) / scale,
                    "vol_ratio_5_20": vol_5 / max(vol_20, np.finfo(float).eps),
                    "phi_hat": phi_hat,
                    "half_life_hat": _half_life(phi_hat),
                    "market_vol": float(data.market_vol[time]),
                    # Retained only for controlled diagnostics, never a feature.
                    "latent_regime": int(data.latent_regime[time]),
                }
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError(f"scenario {data.name} produced no candidate events")
    return frame.sort_values(["time", "pair"], kind="stable").reset_index(drop=True)


def make_folds(config: StudyConfig) -> list[Fold]:
    if config.observations < 2_000:
        train_ends = [700]
        validation_size, calibration_size, selection_size, test_size = 200, 94, 94, 250
    else:
        train_ends = [1_400, 1_900]
        validation_size, calibration_size, selection_size, test_size = 350, 170, 168, 450
    folds = []
    for number, train_end in enumerate(train_ends, start=1):
        validation_start = train_end + config.horizon
        validation_end = validation_start + validation_size
        calibration_start = validation_end + config.horizon
        calibration_end = calibration_start + calibration_size
        selection_start = calibration_end + config.horizon
        selection_end = selection_start + selection_size
        test_start = selection_end + config.horizon
        test_end = test_start + test_size
        if test_end + config.horizon > config.observations:
            raise ValueError("study configuration does not leave a complete test horizon")
        folds.append(
            Fold(
                number,
                train_end,
                validation_start,
                validation_end,
                calibration_start,
                calibration_end,
                selection_start,
                selection_end,
                test_start,
                test_end,
            )
        )
    return folds


def split_events(events: pd.DataFrame, fold: Fold) -> dict[str, pd.DataFrame]:
    return {
        "train": events.loc[events.time < fold.train_end].copy(),
        "validation": events.loc[
            (events.time >= fold.validation_start) & (events.time < fold.validation_end)
        ].copy(),
        "calibration": events.loc[
            (events.time >= fold.calibration_start) & (events.time < fold.calibration_end)
        ].copy(),
        "selection": events.loc[
            (events.time >= fold.selection_start) & (events.time < fold.selection_end)
        ].copy(),
        "test": events.loc[
            (events.time >= fold.test_start) & (events.time < fold.test_end)
        ].copy(),
    }


def _features(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)


def _labels(frame: pd.DataFrame) -> np.ndarray:
    return frame.label.to_numpy(dtype=np.int64, copy=True)


def _targets(frame: pd.DataFrame) -> np.ndarray:
    return frame.net_return.to_numpy(dtype=np.float64, copy=True)


def _lightgbm_regressor(config: StudyConfig, **parameters: Any) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=config.iterations,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=5,
        min_child_samples=60,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=config.seed,
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        verbosity=-1,
        **parameters,
    )


def _catboost_regressor(config: StudyConfig, **parameters: Any) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=config.iterations,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=5.0,
        random_strength=0.2,
        random_seed=config.seed,
        has_time=True,
        allow_writing_files=False,
        thread_count=1,
        verbose=False,
        **parameters,
    )


def _fit_point_models(
    split: dict[str, pd.DataFrame], config: StudyConfig
) -> dict[str, dict[str, Any]]:
    train_x, validation_x = _features(split["train"]), _features(split["validation"])
    calibration_x = _features(split["calibration"])
    selection_x, test_x = _features(split["selection"]), _features(split["test"])
    train_y, validation_y = _targets(split["train"]), _targets(split["validation"])
    calibration_y, test_y = _targets(split["calibration"]), _targets(split["test"])

    models = [
        ("lightgbm_point", _lightgbm_regressor(config, objective="regression_l2")),
        ("catboost_point", _catboost_regressor(config, loss_function="RMSE")),
        ("lightgbm_huber", _lightgbm_regressor(config, objective="huber", alpha=0.9)),
        (
            "catboost_huber",
            _catboost_regressor(config, loss_function="Huber:delta=1.0"),
        ),
    ]
    for lane, model in models:
        if lane.startswith("lightgbm"):
            model.fit(
                train_x,
                train_y,
                eval_set=[(validation_x, validation_y)],
                eval_metric="huber" if lane.endswith("huber") else "l2",
                callbacks=[early_stopping(35, verbose=False), log_evaluation(0)],
            )
        else:
            model.fit(
                train_x,
                train_y,
                eval_set=(validation_x, validation_y),
                early_stopping_rounds=35,
                use_best_model=True,
            )

    output = {}
    for lane, model in models:
        raw_calibration = np.asarray(model.predict(calibration_x), dtype=float)
        bias_adjustment = float(np.mean(calibration_y - raw_calibration))
        calibration_prediction = raw_calibration + bias_adjustment
        selection_prediction = (
            np.asarray(model.predict(selection_x), dtype=float) + bias_adjustment
        )
        test_prediction = np.asarray(model.predict(test_x), dtype=float) + bias_adjustment
        output[lane] = {
            "selection_score": selection_prediction,
            "test_score": test_prediction,
            "diagnostics": {
                "point_bias_adjustment": bias_adjustment,
                "calibration_rmse_raw": math.sqrt(
                    mean_squared_error(calibration_y, raw_calibration)
                ),
                "calibration_rmse": math.sqrt(
                    mean_squared_error(calibration_y, calibration_prediction)
                ),
                "test_rmse": math.sqrt(mean_squared_error(test_y, test_prediction)),
                "test_mae": mean_absolute_error(test_y, test_prediction),
            },
        }
    return output


def _quantile_adjustment(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.quantile(target - prediction[:, index], alpha)
            for index, alpha in enumerate(QUANTILES)
        ]
    )


def _ordered_quantiles(prediction: np.ndarray, adjustment: np.ndarray) -> np.ndarray:
    return np.sort(np.asarray(prediction, dtype=float) + adjustment, axis=1)


def downside_adjusted_quantile_score(prediction: np.ndarray) -> np.ndarray:
    """Rank candidates by median return with a fixed lower-tail penalty."""

    prediction = np.asarray(prediction, dtype=float)
    if prediction.ndim != 2 or prediction.shape[1] != len(QUANTILES):
        raise ValueError("expected one prediction column per configured quantile")
    downside_spread = prediction[:, 1] - prediction[:, 0]
    return prediction[:, 1] - QUANTILE_DOWNSIDE_PENALTY * downside_spread


def _fit_quantile_models(
    split: dict[str, pd.DataFrame], config: StudyConfig
) -> dict[str, dict[str, Any]]:
    train_x, validation_x = _features(split["train"]), _features(split["validation"])
    calibration_x = _features(split["calibration"])
    selection_x, test_x = _features(split["selection"]), _features(split["test"])
    train_y, validation_y = _targets(split["train"]), _targets(split["validation"])
    calibration_y, test_y = _targets(split["calibration"]), _targets(split["test"])

    light_calibration = []
    light_selection = []
    light_test = []
    for alpha in QUANTILES:
        model = _lightgbm_regressor(config, objective="quantile", alpha=float(alpha))
        model.fit(
            train_x,
            train_y,
            eval_set=[(validation_x, validation_y)],
            eval_metric="quantile",
            callbacks=[early_stopping(35, verbose=False), log_evaluation(0)],
        )
        light_calibration.append(model.predict(calibration_x))
        light_selection.append(model.predict(selection_x))
        light_test.append(model.predict(test_x))
    raw_light_calibration = np.column_stack(light_calibration)
    raw_light_selection = np.column_stack(light_selection)
    raw_light_test = np.column_stack(light_test)

    cat = _catboost_regressor(
        config,
        loss_function="MultiQuantile:alpha=0.2,0.5,0.8",
    )
    cat.fit(
        train_x,
        train_y,
        eval_set=(validation_x, validation_y),
        early_stopping_rounds=35,
        use_best_model=True,
    )
    raw_cat_calibration = np.asarray(cat.predict(calibration_x), dtype=float)
    raw_cat_selection = np.asarray(cat.predict(selection_x), dtype=float)
    raw_cat_test = np.asarray(cat.predict(test_x), dtype=float)

    output = {}
    for lane, raw_calibration, raw_selection, raw_test in [
        (
            "lightgbm_quantile",
            raw_light_calibration,
            raw_light_selection,
            raw_light_test,
        ),
        ("catboost_quantile", raw_cat_calibration, raw_cat_selection, raw_cat_test),
    ]:
        crossing_rate = float(np.mean(np.any(np.diff(raw_test, axis=1) < 0.0, axis=1)))
        adjustment = _quantile_adjustment(calibration_y, raw_calibration)
        calibrated = _ordered_quantiles(raw_calibration, adjustment)
        selected = _ordered_quantiles(raw_selection, adjustment)
        tested = _ordered_quantiles(raw_test, adjustment)
        calibration_coverage = [
            float(np.mean(calibration_y <= calibrated[:, index]))
            for index in range(len(QUANTILES))
        ]
        coverage = [float(np.mean(test_y <= tested[:, index])) for index in range(len(QUANTILES))]
        pinball = [
            float(mean_pinball_loss(test_y, tested[:, index], alpha=float(alpha)))
            for index, alpha in enumerate(QUANTILES)
        ]
        output[lane] = {
            "selection_score": downside_adjusted_quantile_score(selected),
            "test_score": downside_adjusted_quantile_score(tested),
            "diagnostics": {
                "quantile_adjustment": adjustment.tolist(),
                "calibration_coverage_20": calibration_coverage[0],
                "calibration_coverage_50": calibration_coverage[1],
                "calibration_coverage_80": calibration_coverage[2],
                "test_coverage_20": coverage[0],
                "test_coverage_50": coverage[1],
                "test_coverage_80": coverage[2],
                "test_pinball_20": pinball[0],
                "test_pinball_50": pinball[1],
                "test_pinball_80": pinball[2],
                "raw_crossing_rate": crossing_rate,
                "test_mean_interval_width": float(np.mean(tested[:, 2] - tested[:, 0])),
                "test_strict_q20_positive_rate": float(np.mean(tested[:, 0] > 0.0)),
            },
        }
    return output


def _class_weights(labels: np.ndarray) -> dict[int, float]:
    counts = np.bincount(labels, minlength=3)
    total = float(counts.sum())
    return {index: total / (3.0 * max(int(count), 1)) for index, count in enumerate(counts)}


def _calibrated_classifier(model: Any, calibration_x: np.ndarray, calibration_y: np.ndarray):
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    except ImportError:
        # scikit-learn 1.5 supports the same disjoint prefit contract through
        # this spelling; FrozenEstimator is used automatically on newer builds.
        calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        calibrated.fit(calibration_x, calibration_y)
    return calibrated


def expected_barrier_score(
    probability: np.ndarray,
    classes: Iterable[int],
    class_payoffs: dict[int, float],
) -> np.ndarray:
    """Convert calibrated class probabilities into expected net return."""

    payoff_vector = np.asarray([class_payoffs[int(value)] for value in classes], dtype=float)
    return np.asarray(probability, dtype=float) @ payoff_vector


def _multiclass_brier(labels: np.ndarray, probability: np.ndarray) -> float:
    observed = np.eye(3, dtype=float)[labels]
    return float(np.mean(np.sum(np.square(probability - observed), axis=1)))


def _expected_calibration_error(
    labels: np.ndarray, probability: np.ndarray, positive_class: int = 2, bins: int = 8
) -> float:
    positive = (labels == positive_class).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        include = (probability >= lower) & (
            probability <= upper if upper == 1.0 else probability < upper
        )
        if not np.any(include):
            continue
        error += float(np.mean(include)) * abs(
            float(np.mean(probability[include])) - float(np.mean(positive[include]))
        )
    return error


def _fit_barrier_models(
    split: dict[str, pd.DataFrame], config: StudyConfig
) -> dict[str, dict[str, Any]]:
    train_x, validation_x = _features(split["train"]), _features(split["validation"])
    calibration_x = _features(split["calibration"])
    selection_x, test_x = _features(split["selection"]), _features(split["test"])
    train_y = _labels(split["train"]) + 1
    validation_y = _labels(split["validation"]) + 1
    calibration_y = _labels(split["calibration"]) + 1
    calibration_return = _targets(split["calibration"])
    test_y = _labels(split["test"]) + 1
    weights = _class_weights(train_y)

    light = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=config.iterations,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=5,
        min_child_samples=60,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        class_weight=weights,
        random_state=config.seed,
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        verbosity=-1,
    )
    light.fit(
        train_x,
        train_y,
        eval_set=[(validation_x, validation_y)],
        eval_metric="multi_logloss",
        callbacks=[early_stopping(35, verbose=False), log_evaluation(0)],
    )
    cat = CatBoostClassifier(
        iterations=config.iterations,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=5.0,
        random_strength=0.2,
        loss_function="MultiClass",
        class_weights=[weights[index] for index in range(3)],
        random_seed=config.seed,
        has_time=True,
        allow_writing_files=False,
        thread_count=1,
        verbose=False,
    )
    cat.fit(
        train_x,
        train_y,
        eval_set=(validation_x, validation_y),
        early_stopping_rounds=35,
        use_best_model=True,
    )

    output = {}
    for lane, model in [("lightgbm_barrier", light), ("catboost_barrier", cat)]:
        calibrated = _calibrated_classifier(model, calibration_x, calibration_y)
        calibration_probability = np.asarray(calibrated.predict_proba(calibration_x), dtype=float)
        selection_probability = np.asarray(calibrated.predict_proba(selection_x), dtype=float)
        test_probability = np.asarray(calibrated.predict_proba(test_x), dtype=float)
        classes = np.asarray(calibrated.classes_, dtype=int)
        win_column = int(np.flatnonzero(classes == 2)[0])
        default_payoffs = {
            0: -config.stop_barrier - config.round_trip_cost,
            1: -config.round_trip_cost,
            2: config.profit_barrier - config.round_trip_cost,
        }
        class_payoffs = {
            int(value): (
                float(np.mean(calibration_return[calibration_y == value]))
                if np.any(calibration_y == value)
                else default_payoffs[int(value)]
            )
            for value in classes
        }
        output[lane] = {
            "selection_score": expected_barrier_score(
                selection_probability, classes, class_payoffs
            ),
            "test_score": expected_barrier_score(test_probability, classes, class_payoffs),
            "diagnostics": {
                "barrier_class_payoffs": class_payoffs,
                "calibration_log_loss": float(
                    log_loss(calibration_y, calibration_probability, labels=classes)
                ),
                "calibration_win_ece": _expected_calibration_error(
                    calibration_y, calibration_probability[:, win_column]
                ),
                "test_log_loss": float(log_loss(test_y, test_probability, labels=classes)),
                "test_multiclass_brier": _multiclass_brier(test_y, test_probability),
                "test_win_ece": _expected_calibration_error(
                    test_y, test_probability[:, win_column]
                ),
                "test_mean_win_probability": float(np.mean(test_probability[:, win_column])),
                "test_win_frequency": float(np.mean(test_y == 2)),
            },
        }
    return output


def simulate_trades(
    frame: pd.DataFrame, score: np.ndarray, threshold: float
) -> tuple[dict[str, float | int], pd.DataFrame]:
    if len(frame) != len(score):
        raise ValueError("score and event lengths differ")
    candidates = frame.copy()
    candidates["score"] = np.asarray(score, dtype=float)
    candidates = candidates.loc[np.isfinite(candidates.score) & (candidates.score >= threshold)]
    busy_until: dict[int, int] = {}
    selected = []
    for row in candidates.sort_values(["time", "pair"], kind="stable").itertuples():
        pair = int(row.pair)
        if int(row.time) <= busy_until.get(pair, -1):
            continue
        busy_until[pair] = int(row.exit_time)
        selected.append(row.Index)
    trades = frame.loc[selected].copy().sort_values(["exit_time", "pair"], kind="stable")
    if trades.empty:
        return {
            "trades": 0,
            "total_net": 0.0,
            "mean_net": 0.0,
            "hit_rate": 0.0,
            "trade_tstat": 0.0,
            "max_drawdown": 0.0,
            "average_duration": 0.0,
        }, trades
    returns = trades.net_return.to_numpy(dtype=float)
    deviation = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    trade_tstat = (
        float(np.mean(returns) / deviation * math.sqrt(len(returns)))
        if deviation > 0
        else 0.0
    )
    equity = np.cumsum(returns)
    running_peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    drawdown = equity - running_peak
    metrics: dict[str, float | int] = {
        "trades": int(len(trades)),
        "total_net": float(np.sum(returns)),
        "mean_net": float(np.mean(returns)),
        "hit_rate": float(np.mean(returns > 0.0)),
        "trade_tstat": trade_tstat,
        "max_drawdown": float(abs(np.min(np.minimum(drawdown, 0.0)))),
        "average_duration": float(np.mean(trades.duration)),
    }
    trades["equity"] = equity
    return metrics, trades


def choose_threshold(
    frame: pd.DataFrame,
    score: np.ndarray,
    min_trades: int,
    minimum_threshold: float = 0.0,
) -> tuple[float, dict[str, float | int]]:
    finite = np.asarray(score, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return minimum_threshold, simulate_trades(frame, score, minimum_threshold)[0]
    grid = np.quantile(finite, [0.0, 0.20, 0.40, 0.60, 0.75, 0.85, 0.90])
    thresholds = sorted({float(max(minimum_threshold, value)) for value in grid})
    best: tuple[tuple[float, float, int], float, dict[str, float | int]] | None = None
    for threshold in thresholds:
        metrics, _ = simulate_trades(frame, score, threshold)
        if int(metrics["trades"]) < min_trades:
            continue
        objective = (
            float(metrics["trade_tstat"]),
            float(metrics["total_net"]),
            int(metrics["trades"]),
        )
        if best is None or objective > best[0]:
            best = (objective, threshold, metrics)
    if best is None:
        threshold = minimum_threshold
        return threshold, simulate_trades(frame, score, threshold)[0]
    return best[1], best[2]


def _fold_outputs(
    split: dict[str, pd.DataFrame], config: StudyConfig
) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {
        "rolling_z": {
            "selection_score": split["selection"].abs_z.to_numpy(dtype=float),
            "test_score": split["test"].abs_z.to_numpy(dtype=float),
            "diagnostics": {},
        }
    }
    outputs.update(_fit_point_models(split, config))
    outputs.update(_fit_quantile_models(split, config))
    outputs.update(_fit_barrier_models(split, config))
    return outputs


def run_study(
    config: StudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metric_records = []
    diagnostic_records = []
    trade_records = []
    scenario_manifest = {}
    for scenario_name in ["stationary_ou", "regime_switching"]:
        scenario = generate_scenario(scenario_name, config)
        events = make_events(scenario, config)
        scenario_manifest[scenario_name] = {
            "events": int(len(events)),
            "labels": {
                str(int(label)): int(count)
                for label, count in events.label.value_counts().sort_index().items()
            },
            "regime_frequency": {
                str(int(regime)): float(frequency)
                for regime, frequency in events.latent_regime.value_counts(normalize=True)
                .sort_index()
                .items()
            },
        }
        for fold in make_folds(config):
            split = split_events(events, fold)
            if any(len(frame) < 50 for frame in split.values()):
                raise RuntimeError(f"insufficient events in {scenario_name} fold {fold.fold}")
            outputs = _fold_outputs(split, config)
            for lane in LANE_ORDER:
                output = outputs[lane]
                threshold, selection_metrics = choose_threshold(
                    split["selection"],
                    output["selection_score"],
                    config.min_selection_trades,
                    config.candidate_z if lane == "rolling_z" else 0.0,
                )
                metrics, trades = simulate_trades(split["test"], output["test_score"], threshold)
                metric_records.append(
                    {
                        "scenario": scenario_name,
                        "fold": fold.fold,
                        "lane": lane,
                        "threshold": threshold,
                        "test_candidates": len(split["test"]),
                        "selection_rate": int(metrics["trades"]) / len(split["test"]),
                        **metrics,
                        "selection_trades": selection_metrics["trades"],
                        "selection_tstat": selection_metrics["trade_tstat"],
                    }
                )
                diagnostic_records.append(
                    {
                        "scenario": scenario_name,
                        "fold": fold.fold,
                        "lane": lane,
                        **output["diagnostics"],
                    }
                )
                if not trades.empty:
                    trades = trades.assign(scenario=scenario_name, fold=fold.fold, lane=lane)
                    trade_records.append(
                        trades.loc[
                            :,
                            [
                                "scenario",
                                "fold",
                                "lane",
                                "pair",
                                "time",
                                "exit_time",
                                "duration",
                                "label",
                                "net_return",
                                "latent_regime",
                            ],
                        ]
                    )

    metrics = pd.DataFrame.from_records(metric_records)
    diagnostics = pd.DataFrame.from_records(diagnostic_records)
    trades = pd.concat(trade_records, ignore_index=True) if trade_records else pd.DataFrame()
    manifest = {
        "config": asdict(config),
        "folds": [asdict(fold) for fold in make_folds(config)],
        "features": FEATURE_COLUMNS,
        "quantiles": QUANTILES.tolist(),
        "quantile_downside_penalty": QUANTILE_DOWNSIDE_PENALTY,
        "lanes": LANE_ORDER,
        "protocol": {
            "block_order": [
                "fit",
                "early_stop_validation",
                "predictive_calibration",
                "policy_selection",
                "untouched_test",
            ],
            "purge_gap_observations": config.horizon,
            "overlap_rule": "no overlapping positions within a pair",
            "trade_metric": "descriptive t-statistic over normalized net trade returns",
        },
        "scenario_design": {
            "stationary_ou": "Gaussian homogeneous mean reversion negative control",
            "regime_switching": (
                "observable volatility proxy, latent persistence states, and unit-variance "
                "Student-t(4) idiosyncratic shocks in stressed states"
            ),
        },
        "scenarios": scenario_manifest,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "catboost": catboost.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    return metrics, diagnostics, trades, manifest


def aggregate_metrics(metrics: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    records = []
    for scenario in metrics.scenario.drop_duplicates():
        for lane in LANE_ORDER:
            selected_metrics = metrics.loc[
                (metrics.scenario == scenario) & (metrics.lane == lane)
            ]
            lane_trades = trades.loc[
                (trades.scenario == scenario) & (trades.lane == lane)
            ].copy()
            lane_trades = lane_trades.sort_values(["exit_time", "pair"], kind="stable")
            returns = lane_trades.net_return.to_numpy(dtype=float)
            deviation = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
            equity = np.cumsum(returns)
            running_peak = (
                np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
                if len(equity)
                else np.asarray([], dtype=float)
            )
            records.append(
                {
                    "scenario": scenario,
                    "lane": lane,
                    "folds": int(selected_metrics.fold.nunique()),
                    "trades": int(len(returns)),
                    "total_net": float(np.sum(returns)) if len(returns) else 0.0,
                    "mean_net": float(np.mean(returns)) if len(returns) else 0.0,
                    "hit_rate": float(np.mean(returns > 0.0)) if len(returns) else 0.0,
                    "trade_tstat": (
                        float(np.mean(returns) / deviation * math.sqrt(len(returns)))
                        if deviation > 0
                        else 0.0
                    ),
                    "max_drawdown": (
                        float(abs(np.min(np.minimum(equity - running_peak, 0.0))))
                        if len(equity)
                        else 0.0
                    ),
                    "average_duration": (
                        float(np.mean(lane_trades.duration)) if len(returns) else 0.0
                    ),
                    "selection_rate": float(selected_metrics.selection_rate.mean()),
                }
            )
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["scenario", "lane"])
        .reset_index(drop=True)
    )


def save_plots(
    aggregate: pd.DataFrame,
    diagnostics: pd.DataFrame,
    trades: pd.DataFrame,
    output_directory: Path,
) -> None:
    colors = dict(zip(LANE_ORDER, plt.cm.tab10.colors))
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for row, scenario in enumerate(["stationary_ou", "regime_switching"]):
        axis = axes[row, 0]
        for lane in LANE_ORDER:
            selected = trades.loc[
                (trades.scenario == scenario) & (trades.lane == lane)
            ].sort_values(["exit_time", "pair"], kind="stable")
            if selected.empty:
                continue
            axis.plot(
                np.arange(1, len(selected) + 1),
                selected.net_return.cumsum(),
                label=lane,
                color=colors[lane],
                linewidth=1.6,
            )
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_title(f"{scenario.replace('_', ' ').title()}: untouched test trades")
        axis.set_xlabel("closed trades")
        axis.set_ylabel("cumulative normalized net return")
        axis.grid(alpha=0.2)
        if row == 0:
            axis.legend(fontsize=8, ncol=2)

        bar_axis = axes[row, 1]
        values = (
            aggregate.loc[aggregate.scenario == scenario]
            .set_index("lane")
            .reindex(LANE_ORDER)
        )
        positions = np.arange(len(values))
        bar_axis.bar(
            positions,
            values.trade_tstat,
            color=[colors[lane] for lane in values.index],
        )
        bar_axis.axhline(0.0, color="black", linewidth=0.7)
        bar_axis.set_xticks(
            positions,
            [lane.replace("_", "\n") for lane in values.index],
            fontsize=7,
        )
        bar_axis.set_ylabel("trade-return t-statistic")
        bar_axis.set_title("Risk-adjusted trade evidence (not annualized Sharpe)")
        bar_axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Hypercube controlled stat-arb study — fit, calibration, selection, and untouched tests",
        fontsize=14,
    )
    figure.savefig(output_directory / "statarb_ml_results.png", dpi=170)
    plt.close(figure)

    calibration = diagnostics.loc[
        diagnostics.lane.isin(
            ["lightgbm_quantile", "catboost_quantile", "lightgbm_barrier", "catboost_barrier"]
        )
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    quantile = calibration.loc[calibration.lane.str.endswith("quantile")]
    grouped = quantile.groupby(["scenario", "lane"])[
        ["test_coverage_20", "test_coverage_50", "test_coverage_80"]
    ].mean()
    labels = [f"{scenario}\n{lane}" for scenario, lane in grouped.index]
    positions = np.arange(len(grouped))
    width = 0.22
    for offset, column, target in [
        (-width, "test_coverage_20", 0.2),
        (0.0, "test_coverage_50", 0.5),
        (width, "test_coverage_80", 0.8),
    ]:
        axes[0].bar(positions + offset, grouped[column], width=width, label=f"target {target:.1f}")
    axes[0].set_xticks(positions, labels, fontsize=8)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("empirical test coverage")
    axes[0].set_title("Marginal quantile calibration")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    classifiers = calibration.loc[calibration.lane.str.endswith("barrier")]
    grouped_classifier = classifiers.groupby(["scenario", "lane"])["test_win_ece"].mean()
    classifier_labels = [f"{scenario}\n{lane}" for scenario, lane in grouped_classifier.index]
    axes[1].bar(np.arange(len(grouped_classifier)), grouped_classifier)
    axes[1].set_xticks(np.arange(len(grouped_classifier)), classifier_labels, fontsize=8)
    axes[1].set_ylabel("expected calibration error")
    axes[1].set_title("Calibrated probability of profit-barrier hit")
    axes[1].grid(axis="y", alpha=0.2)
    figure.savefig(output_directory / "statarb_ml_calibration.png", dpi=170)
    plt.close(figure)


def write_results(
    output_directory: Path,
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    trades: pd.DataFrame,
    manifest: dict[str, Any],
) -> pd.DataFrame:
    output_directory.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_metrics(metrics, trades)
    metrics.to_csv(output_directory / "fold_metrics.csv", index=False)
    diagnostics.to_csv(output_directory / "calibration_diagnostics.csv", index=False)
    aggregate.to_csv(output_directory / "summary.csv", index=False)
    with (output_directory / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    save_plots(aggregate, diagnostics, trades, output_directory)
    return aggregate
