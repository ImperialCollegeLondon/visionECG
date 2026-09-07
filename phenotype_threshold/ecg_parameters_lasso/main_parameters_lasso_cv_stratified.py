#!/usr/bin/env python

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from loader_parameters_lasso import ParametersLassoDataModule
from model_lasso import compute_metrics, cv_grid_search_with_external_val


def filter_csv_by_column(csv_path: str, filter_column: str, filter_value: int,
                         logger: logging.Logger) -> Tuple[str, int, int]:
    df = pd.read_csv(csv_path)
    original = len(df)
    if filter_column not in df.columns:
        raise ValueError(f"Filter column '{filter_column}' not found. Available: {list(df.columns)}")
    if df[filter_column].isna().any():
        df = df.dropna(subset=[filter_column])
    df_filt = df[df[filter_column] == filter_value].copy()
    if len(df_filt) == 0:
        raise ValueError(f"No samples with {filter_column}={filter_value}. "
                         f"Unique values: {sorted(df[filter_column].unique())}")
    base_dir = os.path.dirname(csv_path) or '.'
    base_name = os.path.basename(csv_path).replace('.csv', '')
    out = os.path.join(base_dir, f"{base_name}_filtered_{filter_column}{filter_value}.csv")
    df_filt.to_csv(out, index=False)
    logger.info(f"{csv_path}: {len(df_filt)}/{original} kept after {filter_column}={filter_value} → {out}")
    return out, original, len(df_filt)


def setup_logging(checkpoint_dir: str) -> logging.Logger:
    log_dir = os.path.join(checkpoint_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')

    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")
    return logger


def save_predictions(model, X, y, eids, output_path: str, logger: logging.Logger, dataset_name: str = ""):
    probs = model.predict_proba(X)
    pd.DataFrame({'eid': eids, 'true_label': y, 'predicted_probability': probs}).to_csv(output_path, index=False)
    logger.info(f"Saved {len(probs)} predictions ({dataset_name}) → {output_path}")


def train_lasso_model_cv_stratified(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = setup_logging(args.checkpoint_dir)
    logger.info(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"Sex filter={args.sex} ({'Female' if args.sex == 0 else 'Male'}), "
                f"label={args.label_column}, threshold={args.threshold} ({args.threshold_direction}), "
                f"n_folds={args.n_folds}, alpha=[{args.alpha_min},{args.alpha_max}] n={args.n_alphas}, "
                f"best_metric={args.best_metric}, imbalance={args.imbalance}")

    train_csv_filt, _, _ = filter_csv_by_column(args.train_csv, 'Sex', args.sex, logger)
    val_csv_filt, _, _ = filter_csv_by_column(args.val_csv, 'Sex', args.sex, logger)

    data_module = ParametersLassoDataModule(
        train_csv=train_csv_filt,
        val_csv=val_csv_filt,
        ecg_parameters_path=args.ecg_parameters_path,
        label_column=args.label_column,
        parameter_cols=[c.strip() for c in args.parameter_cols.split(',') if c.strip()],
        atrial_features=[c.strip() for c in args.atrial_features.split(',') if c.strip()],
        threshold=args.threshold,
        threshold_direction=args.threshold_direction,
    )
    X_train, y_train, X_val, y_val, eids_train, eids_val = data_module.prepare_data()
    feature_names = data_module.get_feature_names()

    alpha_grid = np.logspace(np.log10(max(args.alpha_min, 1e-4)), np.log10(args.alpha_max), args.n_alphas)
    if args.alpha_min <= 0:
        alpha_grid = np.concatenate([[0.0], alpha_grid[1:]])

    (best_alpha, median_model, fold_models_at_best, all_fold_indices,
     cv_results_detailed, cv_results_summary, median_fold_info) = cv_grid_search_with_external_val(
        X_train, y_train, X_val, y_val, alpha_grid,
        best_metric=args.best_metric, n_folds=args.n_folds,
        feature_names=feature_names, imbalance=args.imbalance, random_state=args.seed,
    )

    cv_results_detailed.to_csv(os.path.join(args.checkpoint_dir, "cv_results_detailed.csv"), index=False)
    at_best = cv_results_detailed[
        np.isclose(cv_results_detailed['alpha'].astype(float), float(best_alpha))
    ]
    at_best.to_csv(os.path.join(args.checkpoint_dir, "cv_results_at_best_alpha.csv"), index=False)
    cv_results_summary.to_csv(os.path.join(args.checkpoint_dir, "cv_results_summary.csv"), index=False)
    pd.DataFrame([median_fold_info]).to_csv(
        os.path.join(args.checkpoint_dir, "median_fold_info.csv"), index=False)
    with open(os.path.join(args.checkpoint_dir, "median_fold_idx.txt"), 'w') as f:
        f.write(str(int(median_fold_info['median_fold_idx'])))

    coeffs_df = median_model.get_coefficients_df(feature_names)
    coeffs_df.to_csv(os.path.join(args.checkpoint_dir, "feature_importance.csv"), index=False)
    logger.info(f"Active features: {len(median_model.get_active_features(feature_names))}/{len(feature_names)}")

    for fold_idx, fold_info in enumerate(all_fold_indices):
        train_idx = fold_info['train_idx']
        val_idx = fold_info['val_idx']
        fold_model = fold_models_at_best[fold_idx]
        save_predictions(fold_model, X_train[train_idx], y_train[train_idx], eids_train[train_idx],
                         os.path.join(args.checkpoint_dir, f"fold_{fold_idx}_train.csv"),
                         logger, f"fold_{fold_idx}_train")
        save_predictions(fold_model, X_train[val_idx], y_train[val_idx], eids_train[val_idx],
                         os.path.join(args.checkpoint_dir, f"fold_{fold_idx}_internal_val.csv"),
                         logger, f"fold_{fold_idx}_internal_val")
        save_predictions(fold_model, X_val, y_val, eids_val,
                         os.path.join(args.checkpoint_dir, f"fold_{fold_idx}_external_test.csv"),
                         logger, f"fold_{fold_idx}_external_test")

    median_model.save(os.path.join(args.checkpoint_dir, "median_model.pkl"))

    save_predictions(median_model, X_train, y_train, eids_train,
                     os.path.join(args.checkpoint_dir, "train_predictions.csv"), logger, "train (all)")
    save_predictions(median_model, X_val, y_val, eids_val,
                     os.path.join(args.checkpoint_dir, "val_predictions.csv"), logger, "val (all)")

    train_metrics = compute_metrics(y_train, median_model.predict_proba(X_train))
    val_metrics = compute_metrics(y_val, median_model.predict_proba(X_val))
    logger.info(f"Median model (alpha={best_alpha:.4f}, fold={median_fold_info['median_fold_idx']}) "
                f"train AUC={train_metrics['auc']:.4f}, val AUC={val_metrics['auc']:.4f}")

    for p in (train_csv_filt, val_csv_filt):
        try:
            os.remove(p)
        except OSError as e:
            logger.warning(f"Could not remove {p}: {e}")


if __name__ == "__main__":
    dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description="Sex-stratified LASSO on 16 ECG parameters, 5-fold CV")
    parser.add_argument("--sex", type=int, required=True, choices=[0, 1],
                        help="0=female, 1=male")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--ecg_parameters_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--label_column", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--parameter_cols", type=str, required=True,
                        help="Comma-separated ECG parameter column names.")
    parser.add_argument("--atrial_features", type=str, required=True,
                        help="Comma-separated subset of parameter_cols to fill NaN with 0 (AF).")
    parser.add_argument("--threshold_direction", type=str, default="greater_than",
                        choices=["less_than", "greater_than"])
    parser.add_argument("--best_metric", type=str, default="loss", choices=["loss", "auc"])
    parser.add_argument("--imbalance", type=str, default="weight",
                        choices=["weight", "smote", "none"])
    parser.add_argument("--output_prefix", type=str, default="")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--alpha_min", type=float, default=0.001)
    parser.add_argument("--alpha_max", type=float, default=10.0)
    parser.add_argument("--n_alphas", type=int, default=101)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    sex_label = f"sex{args.sex}_{'female' if args.sex == 0 else 'male'}"
    args.checkpoint_dir = os.path.join(
        args.checkpoint_dir,
        f"{args.output_prefix}{sex_label}_{args.label_column.replace('/', '_')}"
        f"_{args.threshold_direction}{args.threshold}_cv{args.n_folds}_{dt_string}",
    )
    train_lasso_model_cv_stratified(args)
