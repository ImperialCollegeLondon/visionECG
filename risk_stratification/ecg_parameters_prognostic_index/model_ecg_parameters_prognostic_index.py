#!/usr/bin/env python3
"""Optimize ECG-parameter prognostic models by nested cross-validation."""

from __future__ import annotations

import logging
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from config_ecg_parameters_prognostic_index import (
    EARLY_STOPPING_ROUNDS,
    N_ESTIMATORS_CEILING,
    N_ESTIMATORS_FLOOR,
    OPTUNA_MEDIAN_PRUNER_STARTUP,
    OPTUNA_MEDIAN_PRUNER_WARMUP,
    OPTUNA_STARTUP_TRIALS,
    OUTER_REFIT_ES_HOLDOUT_FRAC,
    XGB_DEVICE,
    XGB_EVAL_METRIC,
    XGB_OBJECTIVE,
    XGB_TREE_METHOD,
)

VALID_OPTUNA_METRICS = ("logloss", "brier", "auc")


def make_xgb_classifier(
    params: Dict, seed: int, n_estimators: int, n_jobs: int,
    use_early_stopping: bool,
) -> xgb.XGBClassifier:
    """Instantiate an XGBClassifier with true-frequency defaults."""
    base = dict(params)
    base["n_estimators"] = int(n_estimators)
    kwargs = dict(
        objective=XGB_OBJECTIVE,
        eval_metric=XGB_EVAL_METRIC,
        tree_method=XGB_TREE_METHOD,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=0,
        device=XGB_DEVICE,
    )
    if use_early_stopping:
        kwargs["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    kwargs.update(base)
    return xgb.XGBClassifier(**kwargs)


def suggest_params(trial: optuna.Trial, search_space: Dict[str, Dict]) -> Dict:
    """Sample hyperparameters from a JSON-defined search space."""
    out: Dict = {}
    for name, spec in search_space.items():
        t = spec["type"]
        if t == "int":
            out[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        elif t == "float":
            out[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        else:
            raise ValueError(f"Unsupported search-space type {t!r} for {name!r}")
    return out


def _score_metric(optuna_metric: str, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if optuna_metric == "logloss":
        return float(log_loss(y_true, np.clip(y_prob, 1e-15, 1.0 - 1e-15)))
    if optuna_metric == "brier":
        return float(brier_score_loss(y_true, y_prob))
    if optuna_metric == "auc":
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    raise ValueError(f"Unknown optuna_metric={optuna_metric!r}")


def _objective(
    trial: optuna.Trial,
    X_train: np.ndarray, y_train: np.ndarray,
    base_seed: int, n_splits: int, n_repeats: int, n_jobs: int,
    optuna_metric: str, search_space: Dict[str, Dict],
) -> float:
    """Inner CV objective on the outer-train partition."""
    params = suggest_params(trial, search_space)
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=base_seed,
    )
    fold_scores: List[float] = []
    fold_aucs: List[float] = []
    fold_loglosses: List[float] = []
    fold_briers: List[float] = []
    best_iters: List[int] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train)):
        X_f_tr, y_f_tr = X_train[tr_idx], y_train[tr_idx]
        X_f_va, y_f_va = X_train[va_idx], y_train[va_idx]

        try:
            X_f_tr2, X_f_es, y_f_tr2, y_f_es = train_test_split(
                X_f_tr, y_f_tr, test_size=OUTER_REFIT_ES_HOLDOUT_FRAC,
                stratify=y_f_tr, random_state=base_seed + fold_idx,
            )
        except ValueError:
            X_f_tr2, X_f_es, y_f_tr2, y_f_es = train_test_split(
                X_f_tr, y_f_tr, test_size=OUTER_REFIT_ES_HOLDOUT_FRAC,
                random_state=base_seed + fold_idx,
            )

        model = make_xgb_classifier(
            params, seed=base_seed, n_estimators=N_ESTIMATORS_CEILING,
            n_jobs=n_jobs, use_early_stopping=True,
        )
        try:
            model.fit(X=X_f_tr2, y=y_f_tr2, eval_set=[(X_f_es, y_f_es)], verbose=False)
        except Exception as exc:
            raise optuna.TrialPruned() from exc

        va_prob = model.predict_proba(X_f_va)[:, 1]
        fold_score = _score_metric(optuna_metric, y_f_va, va_prob)
        fold_scores.append(fold_score)

        if len(np.unique(y_f_va)) >= 2:
            fold_aucs.append(float(roc_auc_score(y_f_va, va_prob)))
            fold_loglosses.append(
                float(log_loss(y_f_va, np.clip(va_prob, 1e-15, 1.0 - 1e-15)))
            )
            fold_briers.append(float(brier_score_loss(y_f_va, va_prob)))

        bi = getattr(model, "best_iteration", None)
        if bi is None:
            bi = model.n_estimators
        best_iters.append(int(bi))

        trial.report(float(np.mean(fold_scores)), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_score = float(np.mean(fold_scores))
    trial.set_user_attr("optuna_metric", optuna_metric)
    trial.set_user_attr("mean_score", mean_score)
    trial.set_user_attr("mean_fold_auc",
                        float(np.mean(fold_aucs)) if fold_aucs else float("nan"))
    trial.set_user_attr("mean_fold_logloss",
                        float(np.mean(fold_loglosses)) if fold_loglosses else float("nan"))
    trial.set_user_attr("mean_fold_brier",
                        float(np.mean(fold_briers)) if fold_briers else float("nan"))
    trial.set_user_attr("best_iterations", best_iters)
    trial.set_user_attr("fold_scores", fold_scores)
    return mean_score


def run_inner_optuna_search(
    X_outer_tr: np.ndarray, y_outer_tr: np.ndarray,
    n_trials: int, n_splits_inner: int, n_repeats_inner: int,
    seed: int, timeout: Optional[int], n_jobs: int,
    optuna_metric: str, search_space: Dict[str, Dict],
    logger: logging.Logger,
) -> Tuple[Dict, pd.DataFrame, optuna.Study, List[int]]:
    """Full Optuna study for one outer fold."""
    if optuna_metric not in VALID_OPTUNA_METRICS:
        raise ValueError(
            f"optuna_metric={optuna_metric!r} not in {VALID_OPTUNA_METRICS}"
        )
    direction = "maximize" if optuna_metric == "auc" else "minimize"

    sampler = optuna.samplers.TPESampler(
        seed=seed, n_startup_trials=OPTUNA_STARTUP_TRIALS,
        multivariate=True, group=True,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=OPTUNA_MEDIAN_PRUNER_STARTUP,
        n_warmup_steps=OPTUNA_MEDIAN_PRUNER_WARMUP,
    )
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

    logger.info(
        f"[Optuna] metric={optuna_metric} direction={direction} n_trials={n_trials} "
        f"inner_splits={n_splits_inner} inner_repeats={n_repeats_inner} "
        f"timeout={timeout}s n_jobs={n_jobs} search_params={list(search_space)}"
    )
    started = time.monotonic()

    def _wrapped(trial: optuna.Trial) -> float:
        return _objective(
            trial, X_outer_tr, y_outer_tr,
            base_seed=seed, n_splits=n_splits_inner, n_repeats=n_repeats_inner,
            n_jobs=n_jobs, optuna_metric=optuna_metric, search_space=search_space,
        )

    def _log_trial(study_instance: optuna.Study, trial: optuna.Trial) -> None:
        elapsed_min = (time.monotonic() - started) / 60.0
        value = "NA" if trial.value is None else f"{trial.value:.4f}"
        try:
            best = f"{study_instance.best_value:.4f}"
        except ValueError:
            best = "NA"
        auc = trial.user_attrs.get("mean_fold_auc", float("nan"))
        ll = trial.user_attrs.get("mean_fold_logloss", float("nan"))
        br = trial.user_attrs.get("mean_fold_brier", float("nan"))
        logger.info(
            f"[Optuna] trial={trial.number + 1}/{n_trials} state={trial.state.name} "
            f"{optuna_metric}={value} auc={auc:.4f} logloss={ll:.4f} brier={br:.4f} "
            f"best={best} elapsed={elapsed_min:.1f}min"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(
            _wrapped, n_trials=n_trials, timeout=timeout,
            gc_after_trial=True, show_progress_bar=False,
            callbacks=[_log_trial],
        )

    completed = sum(1 for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE)
    pruned = sum(1 for t in study.trials
                 if t.state == optuna.trial.TrialState.PRUNED)
    failed = sum(1 for t in study.trials
                 if t.state == optuna.trial.TrialState.FAIL)
    logger.info(f"[Optuna] {completed} complete / {pruned} pruned / {failed} failed")
    logger.info(
        f"[Optuna] Best {optuna_metric}: {study.best_value:.4f} "
        f"(auc={study.best_trial.user_attrs.get('mean_fold_auc', float('nan')):.4f}, "
        f"logloss={study.best_trial.user_attrs.get('mean_fold_logloss', float('nan')):.4f}, "
        f"brier={study.best_trial.user_attrs.get('mean_fold_brier', float('nan')):.4f})"
    )
    logger.info(f"[Optuna] Best params: {study.best_params}")

    best_iters = study.best_trial.user_attrs.get("best_iterations", [])
    trials_df = study.trials_dataframe()
    return dict(study.best_params), trials_df, study, list(map(int, best_iters))


def refit_at_best_params(
    best_params: Dict, X_outer_tr: np.ndarray, y_outer_tr: np.ndarray,
    seed: int, n_jobs: int,
) -> Tuple[xgb.XGBClassifier, int]:
    """Refit one outer-fold model with optimized parameters."""
    try:
        X_inner, X_es, y_inner, y_es = train_test_split(
            X_outer_tr, y_outer_tr, test_size=OUTER_REFIT_ES_HOLDOUT_FRAC,
            stratify=y_outer_tr, random_state=seed,
        )
    except ValueError:
        X_inner, X_es, y_inner, y_es = train_test_split(
            X_outer_tr, y_outer_tr, test_size=OUTER_REFIT_ES_HOLDOUT_FRAC,
            random_state=seed,
        )

    model = make_xgb_classifier(
        best_params, seed=seed, n_estimators=N_ESTIMATORS_CEILING,
        n_jobs=n_jobs, use_early_stopping=True,
    )
    model.fit(X=X_inner, y=y_inner, eval_set=[(X_es, y_es)], verbose=False)
    bi = getattr(model, "best_iteration", None)
    if bi is None:
        bi = model.n_estimators
    return model, max(int(bi), N_ESTIMATORS_FLOOR)


def score_prob(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    """Return sigmoid probabilities P(y=1|X)."""
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
