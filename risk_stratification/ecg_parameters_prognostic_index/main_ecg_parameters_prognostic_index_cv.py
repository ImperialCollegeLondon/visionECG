#!/usr/bin/env python3
"""Run ECG-parameter prognostic nested cross-validation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config_ecg_parameters_prognostic_index import (
    CONFIG_NAME,
    DEFAULT_OPTUNA_METRIC,
    DURATION_COL,
    EVENT_COL,
    EVENT_TYPES,
    FEATURE_MODE_DIR,
    FOLD_AUC_FILENAME,
    INNER_CV_FOLDS,
    INNER_CV_REPEATS,
    LABEL_COL,
    LANDMARK_YEARS_DEFAULT,
    OUTER_CV_FOLDS,
    RANDOM_SEED_DEFAULT,
)
from loader_ecg_parameters_prognostic_index import load_event_cohort, resolve_prefixed_path
from model_ecg_parameters_prognostic_index import (
    VALID_OPTUNA_METRICS,
    refit_at_best_params,
    run_inner_optuna_search,
    score_prob,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def format_landmark_dirname(landmark_years: float, seed: int, suffix: str) -> str:
    y = float(landmark_years)
    if abs(y - round(y)) < 1e-9:
        y_str = f"{int(round(y))}"
    else:
        y_str = f"{y:g}".replace(".", "p")
    name = f"landmark_{y_str}yr_sd{int(seed)}"
    if suffix:
        name = f"{name}_{suffix}"
    return name


def safe_slug(value: str) -> str:
    s = str(value).strip().replace(" ", "_").replace(".", "_").replace("/", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s or "endpoint"


def setup_job_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(f"job_{log_file.stem}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def _hanley_mcneil_auc_se(auc: float, n_pos: int, n_neg: int) -> float:
    if n_pos <= 0 or n_neg <= 0 or not np.isfinite(auc):
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (
        auc * (1.0 - auc)
        + (n_pos - 1) * (q1 - auc * auc)
        + (n_neg - 1) * (q2 - auc * auc)
    ) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def _metrics_row(
    y_true: np.ndarray, prob: np.ndarray, landmark_days: int, landmark_years: float,
) -> Dict:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    mask = np.isfinite(prob) & np.isin(y_true, [0, 1])
    y_true = y_true[mask]
    prob = prob[mask]
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    row: Dict = {
        "landmark_years": float(landmark_years),
        "landmark_days": int(landmark_days),
        "n_total": int(len(y_true)),
        "n_cases": n_pos,
        "n_controls": n_neg,
        "empirical_prevalence": (n_pos / max(len(y_true), 1)),
        "mean_prob": float(np.mean(prob)) if len(prob) else float("nan"),
    }
    if n_pos >= 5 and n_neg >= 5:
        auc = float(roc_auc_score(y_true, prob))
        ll = float(log_loss(y_true, np.clip(prob, 1e-15, 1.0 - 1e-15)))
        br = float(brier_score_loss(y_true, prob))
        row.update({
            "auc": auc,
            "auc_se": _hanley_mcneil_auc_se(auc, n_pos, n_neg),
            "logloss": ll,
            "brier": br,
        })
    else:
        row.update({"auc": float("nan"), "auc_se": float("nan"),
                    "logloss": float("nan"), "brier": float("nan")})
    return row


def _prob_hist_row(prob: np.ndarray, label: np.ndarray, split: str, fold: str) -> Dict:
    prob = np.asarray(prob, dtype=float)
    label = np.asarray(label, dtype=int)
    finite = np.isfinite(prob)
    prob = prob[finite]
    label = label[finite]
    if len(prob) == 0:
        return {"split": split, "fold": fold, "n": 0}
    prev = float(label.mean()) if len(label) else float("nan")
    return {
        "split": split, "fold": fold, "n": int(len(prob)),
        "empirical_prevalence": prev,
        "prob_min": float(np.min(prob)),
        "prob_p05": float(np.quantile(prob, 0.05)),
        "prob_p25": float(np.quantile(prob, 0.25)),
        "prob_median": float(np.median(prob)),
        "prob_mean": float(np.mean(prob)),
        "prob_p75": float(np.quantile(prob, 0.75)),
        "prob_p95": float(np.quantile(prob, 0.95)),
        "prob_max": float(np.max(prob)),
        "abs_mean_prob_minus_prevalence": float(abs(np.mean(prob) - prev)),
    }


def _build_pred_frame(
    df_meta: pd.DataFrame, y_landmark: np.ndarray, prob: np.ndarray,
    fold_tag: str, config_name: str, landmark_years: float, landmark_days: int,
) -> pd.DataFrame:
    id_cols = [c for c in ["patient_id", "eid", "eid_40616"] if c in df_meta.columns]
    out = df_meta[id_cols].copy().reset_index(drop=True)
    out[DURATION_COL] = df_meta[DURATION_COL].to_numpy()
    out[EVENT_COL] = df_meta[EVENT_COL].to_numpy()
    out[LABEL_COL] = np.asarray(y_landmark, dtype=int)
    out["prob"] = np.asarray(prob, dtype=float)
    out["fold"] = fold_tag
    out["config"] = config_name
    out["landmark_years"] = float(landmark_years)
    out["landmark_days"] = int(landmark_days)
    return out


def _make_preprocess_pipeline() -> Tuple[Pipeline, VarianceThreshold]:
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    selector = VarianceThreshold(threshold=0.0)
    return pipe, selector


def run_nested_cv(
    X_train_raw: np.ndarray, y_train: np.ndarray, df_train_meta: pd.DataFrame,
    X_test_raw: np.ndarray, y_test: np.ndarray, df_test_meta: pd.DataFrame,
    feature_names: List[str],
    out_dir: Path, config_name: str, event_alias: str,
    cv_folds: int, n_trials_per_fold: int,
    inner_n_splits: int, inner_n_repeats: int,
    optuna_timeout_sec: int, optuna_metric: str,
    search_space: Dict[str, Dict],
    seed: int, n_jobs: int,
    landmark_years: float, landmark_days: int,
    logger: logging.Logger,
) -> Tuple[List[Dict], List[Dict], np.ndarray, np.ndarray]:
    """Train and evaluate one outer cross-validation loop."""
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fold_metric_rows: List[Dict] = []
    prob_hist_rows: List[Dict] = []
    test_prob_stack = np.full((cv_folds, len(X_test_raw)), np.nan, dtype=float)
    train_oof_prob = np.full(len(X_train_raw), np.nan, dtype=float)
    fold_feature_gains: List[Dict[str, float]] = []

    for pos_idx, (outer_tr_idx, outer_va_idx) in enumerate(
        splitter.split(X_train_raw, y_train)
    ):
        fold_idx = pos_idx + 1
        fold_tag = f"fold_{fold_idx}"
        logger.info(f"===== [Outer fold {fold_idx}/{cv_folds}] {event_alias} =====")

        try:
            X_ov_tr_raw = X_train_raw[outer_tr_idx]
            X_ov_va_raw = X_train_raw[outer_va_idx]
            y_ov_tr = y_train[outer_tr_idx]
            y_ov_va = y_train[outer_va_idx]

            pipe, selector = _make_preprocess_pipeline()
            X_ov_tr = pipe.fit_transform(X_ov_tr_raw).astype(np.float64)
            X_ov_va = pipe.transform(X_ov_va_raw).astype(np.float64)
            X_te = pipe.transform(X_test_raw).astype(np.float64)
            X_ov_tr = selector.fit_transform(X_ov_tr)
            X_ov_va = selector.transform(X_ov_va)
            X_te = selector.transform(X_te)
            kept_names = [feature_names[i] for i in selector.get_support(indices=True)]

            logger.info(f"[fold {fold_idx}] running inner Optuna search "
                        f"(n_trials={n_trials_per_fold}, metric={optuna_metric}) ...")
            best_params, trials_df, study, inner_best_iters = run_inner_optuna_search(
                X_ov_tr, y_ov_tr,
                n_trials=n_trials_per_fold,
                n_splits_inner=inner_n_splits, n_repeats_inner=inner_n_repeats,
                seed=seed + fold_idx, timeout=optuna_timeout_sec, n_jobs=n_jobs,
                optuna_metric=optuna_metric, search_space=search_space, logger=logger,
            )
            trials_df.to_csv(out_dir / f"{fold_tag}_optuna_study.csv", index=False)

            model, refit_best_iter = refit_at_best_params(
                best_params, X_ov_tr, y_ov_tr, seed=seed + fold_idx, n_jobs=n_jobs,
            )
            logger.info(f"[fold {fold_idx}] refit best_iteration = {refit_best_iter}")

            tr_prob = score_prob(model, X_ov_tr)
            va_prob = score_prob(model, X_ov_va)
            te_prob = score_prob(model, X_te)

            train_oof_prob[outer_va_idx] = va_prob
            test_prob_stack[pos_idx] = te_prob

            _build_pred_frame(
                df_train_meta.iloc[outer_tr_idx].reset_index(drop=True),
                y_train[outer_tr_idx], tr_prob,
                fold_tag=fold_tag, config_name=config_name,
                landmark_years=landmark_years, landmark_days=landmark_days,
            ).to_csv(out_dir / f"{fold_tag}_train_pred.csv", index=False)

            _build_pred_frame(
                df_train_meta.iloc[outer_va_idx].reset_index(drop=True),
                y_train[outer_va_idx], va_prob,
                fold_tag=fold_tag, config_name=config_name,
                landmark_years=landmark_years, landmark_days=landmark_days,
            ).to_csv(out_dir / f"{fold_tag}_val_pred.csv", index=False)

            _build_pred_frame(
                df_test_meta.reset_index(drop=True),
                y_test, te_prob,
                fold_tag=fold_tag, config_name=config_name,
                landmark_years=landmark_years, landmark_days=landmark_days,
            ).to_csv(out_dir / f"{fold_tag}_external_test_pred.csv", index=False)

            try:
                joblib.dump(model, out_dir / f"{fold_tag}_model.pkl")
                joblib.dump(
                    {"pipeline": pipe, "variance_selector": selector,
                     "feature_names": kept_names},
                    out_dir / f"{fold_tag}_preprocess.pkl",
                )
            except Exception as exc:
                logger.warning(
                    f"[fold {fold_idx}] pickle failed: {type(exc).__name__}: {exc}"
                )

            with open(out_dir / f"{fold_tag}_optuna_best_params.json", "w") as f:
                json.dump({
                    "event_alias": event_alias,
                    "fold": fold_idx,
                    "config": config_name,
                    "optuna_metric": optuna_metric,
                    "best_score": float(study.best_value),
                    "best_mean_fold_auc": float(
                        study.best_trial.user_attrs.get("mean_fold_auc", float("nan"))
                    ),
                    "best_mean_fold_logloss": float(
                        study.best_trial.user_attrs.get("mean_fold_logloss", float("nan"))
                    ),
                    "best_mean_fold_brier": float(
                        study.best_trial.user_attrs.get("mean_fold_brier", float("nan"))
                    ),
                    "best_params": best_params,
                    "refit_best_iteration": int(refit_best_iter),
                    "inner_best_iterations_per_fold": inner_best_iters,
                    "n_trials": int(n_trials_per_fold),
                    "n_splits_inner": int(inner_n_splits),
                    "n_repeats_inner": int(inner_n_repeats),
                    "seed": int(seed + fold_idx),
                    "n_features": int(len(kept_names)),
                    "selected_feature_names": list(kept_names),
                }, f, indent=2)

            for dataset_name, y_ref, prob_ref in [
                ("train", y_train[outer_tr_idx], tr_prob),
                ("val", y_train[outer_va_idx], va_prob),
                ("external_test", y_test, te_prob),
            ]:
                row = _metrics_row(y_ref, prob_ref, landmark_days, landmark_years)
                fold_metric_rows.append({
                    "fold": fold_idx, "config": config_name,
                    "dataset": dataset_name, **row,
                })
                prob_hist_rows.append(
                    _prob_hist_row(prob_ref, y_ref, dataset_name, fold_tag)
                )

            gain = model.get_booster().get_score(importance_type="gain")
            fold_feature_gains.append({
                kept_names[i]: float(gain.get(f"f{i}", 0.0))
                for i in range(len(kept_names))
            })

            logger.info(
                f"[fold {fold_idx}] AUC train={fold_metric_rows[-3]['auc']:.4f} "
                f"val={fold_metric_rows[-2]['auc']:.4f} "
                f"ext={fold_metric_rows[-1]['auc']:.4f}  "
                f"mean(prob_ext)={fold_metric_rows[-1]['mean_prob']:.4f} "
                f"(prevalence={fold_metric_rows[-1]['empirical_prevalence']:.4f})"
            )
        except Exception as exc:
            logger.error(f"[fold {fold_idx}] failed: {type(exc).__name__}: {exc}")
            import traceback
            logger.error(traceback.format_exc())
            test_prob_stack[pos_idx] = np.nan

    pd.DataFrame(fold_metric_rows).to_csv(out_dir / FOLD_AUC_FILENAME, index=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ensemble_test_prob = np.nanmean(test_prob_stack, axis=0)

    ens_df = _build_pred_frame(
        df_test_meta.reset_index(drop=True), y_test, ensemble_test_prob,
        fold_tag="ensemble", config_name=config_name,
        landmark_years=landmark_years, landmark_days=landmark_days,
    )
    ens_df.to_csv(out_dir / "ensemble_test_pred.csv", index=False)

    final_test_df = ens_df.copy()
    final_test_df["fold"] = "final"
    final_test_df["note"] = "ensemble_of_5_nested_folds"
    final_test_df.to_csv(out_dir / "final_test_pred.csv", index=False)

    final_train_df = _build_pred_frame(
        df_train_meta.reset_index(drop=True), y_train, train_oof_prob,
        fold_tag="final", config_name=config_name,
        landmark_years=landmark_years, landmark_days=landmark_days,
    )
    final_train_df["note"] = "oof_predictions_from_5_nested_folds"
    final_train_df.to_csv(out_dir / "final_train_pred.csv", index=False)

    prob_hist_rows.append(_prob_hist_row(
        ensemble_test_prob, y_test, "external_test", "ensemble",
    ))
    prob_hist_rows.append(_prob_hist_row(
        train_oof_prob, y_train, "train_oof", "ensemble",
    ))
    pd.DataFrame(prob_hist_rows).to_csv(out_dir / "diagnostic_prob_hist.csv", index=False)

    if fold_feature_gains:
        all_features = sorted({f for d in fold_feature_gains for f in d})
        fi_rows = []
        for feat in all_features:
            gains = [d.get(feat, 0.0) for d in fold_feature_gains]
            fi_rows.append({
                "feature": feat, "mean_gain": float(np.mean(gains)),
                "std_gain": float(np.std(gains)),
            })
        (pd.DataFrame(fi_rows).sort_values("mean_gain", ascending=False)
           .to_csv(out_dir / "feature_importance.csv", index=False))

    ens_row = _metrics_row(y_test, ensemble_test_prob, landmark_days, landmark_years)
    logger.info(
        f"[ensemble] external test AUC={ens_row['auc']:.4f} "
        f"logloss={ens_row['logloss']:.4f} brier={ens_row['brier']:.4f} "
        f"mean(prob)={ens_row['mean_prob']:.4f} "
        f"(prevalence={ens_row['empirical_prevalence']:.4f})"
    )
    return fold_metric_rows, prob_hist_rows, ensemble_test_prob, train_oof_prob


def process_single_event(job_params) -> Dict:
    (
        event_alias, root_output_dir,
        main_table_file, train_file, test_file,
        landmark_years,
        cv_folds, n_trials_per_fold,
        inner_n_splits, inner_n_repeats,
        optuna_timeout_sec, optuna_metric,
        search_space,
        seed, n_jobs_xgb,
        feature_cols, config_name,
    ) = job_params

    landmark_days = int(round(landmark_years * 365.25))
    safe_event_name = safe_slug(event_alias)
    event_dir = Path(root_output_dir) / safe_event_name / FEATURE_MODE_DIR
    out_dir = event_dir / config_name
    out_dir.mkdir(parents=True, exist_ok=True)

    slurm_suffix = (
        f"_slurm{os.environ['SLURM_JOB_ID']}"
        if os.environ.get("SLURM_JOB_ID") else ""
    )
    log_file = out_dir / (
        f"{safe_event_name}_{FEATURE_MODE_DIR}_{config_name}{slurm_suffix}.log"
    )
    logger = setup_job_logger(log_file)

    try:
        logger.info(
            f"Starting job: event={event_alias}, config={config_name}, "
            f"landmark={landmark_years}yr, optuna_metric={optuna_metric}, "
            f"n_trials_per_fold={n_trials_per_fold}, cv_folds={cv_folds}, "
            f"inner_splits={inner_n_splits}, inner_repeats={inner_n_repeats}, "
            f"n_features={len(feature_cols)}"
        )

        df_train, df_test, X_train_raw, X_test_raw, feature_names = load_event_cohort(
            main_table_file=main_table_file, train_file=train_file, test_file=test_file,
            event_alias=event_alias, landmark_days=landmark_days,
            feature_cols=feature_cols,
            logger=logger,
        )
        y_train = df_train[LABEL_COL].to_numpy(dtype=int)
        y_test = df_test[LABEL_COL].to_numpy(dtype=int)

        pd.DataFrame([
            {"dataset": "train", "n_total": len(y_train),
             "n_cases": int(y_train.sum()), "n_controls": int((y_train == 0).sum()),
             "landmark_days": int(landmark_days), "landmark_years": float(landmark_years)},
            {"dataset": "test", "n_total": len(y_test),
             "n_cases": int(y_test.sum()), "n_controls": int((y_test == 0).sum()),
             "landmark_days": int(landmark_days), "landmark_years": float(landmark_years)},
        ]).to_csv(out_dir / "landmark_label_summary.csv", index=False)

        run_nested_cv(
            X_train_raw=X_train_raw, y_train=y_train, df_train_meta=df_train,
            X_test_raw=X_test_raw, y_test=y_test, df_test_meta=df_test,
            feature_names=feature_names,
            out_dir=out_dir, config_name=config_name, event_alias=event_alias,
            cv_folds=cv_folds, n_trials_per_fold=n_trials_per_fold,
            inner_n_splits=inner_n_splits, inner_n_repeats=inner_n_repeats,
            optuna_timeout_sec=optuna_timeout_sec, optuna_metric=optuna_metric,
            search_space=search_space,
            seed=seed, n_jobs=n_jobs_xgb,
            landmark_years=landmark_years, landmark_days=landmark_days,
            logger=logger,
        )

        logger.info(f"== Completed event: {event_alias} ==")
        return {"status": "success", "event_alias": event_alias, "out_dir": str(out_dir)}

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error(f"Job failed: {event_alias}")
        logger.error(f"Error: {error_msg}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "failed", "event_alias": event_alias, "error": error_msg}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="XGBoost + Optuna nested 5-fold CV for the ECG-parameters prognostic index."
    )
    p.add_argument("--abs_pre", type=str, required=True,
                   help="Filesystem prefix prepended to any relative --*_file/--output_root.")
    p.add_argument("--main_table_file", type=str, required=True)
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--test_file", type=str, required=True)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--output_suffix", type=str, required=True,
                   help="Suffix appended to the landmark directory name.")
    p.add_argument("--events", nargs="+", type=str, required=True,
                   choices=EVENT_TYPES + ["all"],
                   help=f"Event(s) to run. 'all' expands to {EVENT_TYPES}.")
    p.add_argument("--ecg_feature_cols", nargs="+", type=str, required=True,
                   help="ECG-parameter column names (bash supplies the full list).")
    p.add_argument("--search_space_json", type=str, required=True,
                   help="JSON string mapping hyperparameter name -> "
                        "{type: int|float, low: <n>, high: <n>, log: <bool>}.")
    p.add_argument("--n_trials_per_fold", type=int, required=True,
                   help="Optuna trials for the inner search of EACH outer fold.")
    p.add_argument("--optuna_timeout_sec", type=int, required=True,
                   help="Wall-clock cap per outer fold's Optuna study.")
    p.add_argument("--landmark_years", type=float, default=LANDMARK_YEARS_DEFAULT)
    p.add_argument("--cv_folds", type=int, default=OUTER_CV_FOLDS)
    p.add_argument("--inner_n_splits", type=int, default=INNER_CV_FOLDS)
    p.add_argument("--inner_n_repeats", type=int, default=INNER_CV_REPEATS)
    p.add_argument("--optuna_metric", type=str, default=DEFAULT_OPTUNA_METRIC,
                   choices=list(VALID_OPTUNA_METRICS))
    p.add_argument("--random_seed", type=int, default=RANDOM_SEED_DEFAULT)
    p.add_argument("--max_workers", type=int, default=1,
                   help="Parallel events (multiprocessing.Pool). 1 = serial.")
    p.add_argument("--n_jobs_xgb", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    for path_arg in ("main_table_file", "train_file", "test_file", "output_root"):
        setattr(args, path_arg,
                resolve_prefixed_path(args.abs_pre, getattr(args, path_arg)))

    events = EVENT_TYPES if (len(args.events) == 1 and args.events[0] == "all") else args.events
    for ev in events:
        if ev not in EVENT_TYPES:
            raise ValueError(f"Unknown event {ev!r}; choose from {EVENT_TYPES}.")

    feature_cols = list(args.ecg_feature_cols)
    if not feature_cols:
        raise ValueError("--ecg_feature_cols must be a non-empty list.")

    try:
        search_space = json.loads(args.search_space_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--search_space_json is not valid JSON: {exc}") from exc
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("--search_space_json must decode to a non-empty JSON object.")

    landmark_dirname = format_landmark_dirname(
        args.landmark_years, args.random_seed, args.output_suffix,
    )
    root_output_dir = Path(args.output_root) / landmark_dirname
    root_output_dir.mkdir(parents=True, exist_ok=True)
    run_config = dict(vars(args))
    run_config["config_name"] = CONFIG_NAME
    run_config["resolved_feature_cols"] = list(feature_cols)
    run_config["n_features"] = int(len(feature_cols))
    run_config["search_space"] = search_space
    with open(root_output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2, sort_keys=True, default=str)

    print(f"Root output dir : {root_output_dir}")
    print(f"Events          : {', '.join(events)}")
    print(f"Features        : {len(feature_cols)}")
    print(f"Config subdir   : {CONFIG_NAME}")
    print(f"CV              : outer={args.cv_folds}, inner={args.inner_n_splits}x{args.inner_n_repeats}")
    print(f"Optuna          : {args.n_trials_per_fold} trials/fold, "
          f"timeout={args.optuna_timeout_sec}s, metric={args.optuna_metric}")
    print(f"Search space    : {list(search_space)}")
    print(f"Landmark years  : {args.landmark_years}")
    print(f"Random seed     : {args.random_seed}")

    jobs = []
    for event_alias in events:
        jobs.append((
            event_alias, root_output_dir,
            args.main_table_file, args.train_file, args.test_file,
            float(args.landmark_years),
            int(args.cv_folds), int(args.n_trials_per_fold),
            int(args.inner_n_splits), int(args.inner_n_repeats),
            int(args.optuna_timeout_sec), str(args.optuna_metric),
            search_space,
            int(args.random_seed), int(args.n_jobs_xgb),
            list(feature_cols), str(CONFIG_NAME),
        ))

    if args.max_workers == 1:
        results = [process_single_event(j) for j in jobs]
    else:
        with Pool(processes=min(args.max_workers, len(jobs))) as pool:
            results = pool.map(process_single_event, jobs)

    successes = sum(1 for r in results if r["status"] == "success")
    failures = sum(1 for r in results if r["status"] == "failed")
    print(f"Successful events: {successes}/{len(jobs)}")
    print(f"Failed events:     {failures}/{len(jobs)}")
    for r in results:
        if r["status"] == "failed":
            print(f"  - {r['event_alias']}: {r.get('error', 'Unknown error')}")
    print(f"All outputs saved to: {root_output_dir}")
    if failures > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
