#!/usr/bin/env python
# -*-coding:utf-8 -*-

import argparse
import ast
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from loader_visionecg_xgboost import VisionECGXGBoostDataModule
from model_visionecg_xgboost import (
    compute_metrics,
    final_cv_with_best_params,
    get_feature_importance_df,
    optuna_search_xgb,
)


def _parse_virtual_label_map(raw: Optional[str]) -> dict:
    if not raw:
        return {}
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        raise argparse.ArgumentTypeError(
            f"--virtual_label_map must be a python-literal dict, got {raw!r}: {exc}"
        )
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            f"--virtual_label_map must be a dict, got {type(parsed).__name__}"
        )
    return parsed


def log_label_distribution(df: pd.DataFrame, disease_cols: list, dataset_name: str, logger=None):
    if logger is None:
        return
    logger.info(f"\n{'='*80}\nLABEL DISTRIBUTION: {dataset_name}\n{'='*80}")
    logger.info(f"Total samples: {len(df)}")
    for disease in disease_cols:
        if disease in df.columns:
            count = (df[disease] == 1).sum()
            pct = 100 * count / len(df)
            logger.info(f"  {disease:<20} {count:>8} {pct:>11.2f}%")


def setup_logging(checkpoint_dir: str):
    log_dir = os.path.join(checkpoint_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'visionecg_xgboost_wrapper.log')

    logger = logging.getLogger('visionecg_xgboost')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("="*80)
    logger.info("visionECG-XGBoost multi-label one-vs-rest classification")
    logger.info("="*80)
    logger.info(f"Log file: {log_file}")
    for h in logger.handlers:
        h.flush()
    return logger


def train_one_strategy(
    strategy: str,
    disease_label: str,
    data: dict,
    feature_names: list,
    strategy_checkpoint: str,
    n_folds: int,
    n_trials: int,
    n_repeats: int,
    optuna_timeout_sec: int,
    use_gpu: bool,
    seed: int,
    search_space: dict,
    logger,
):
    """Run Optuna + final CV for one (disease, strategy) combination."""
    logger.info(f"\n{'#'*80}\n### {disease_label} vs rest — strategy={strategy}\n{'#'*80}")

    X_train = data['X_train']; y_train = data['y_train']
    X_iv = data['X_iv']; y_iv = data['y_iv']
    X_et = data['X_et']; y_et = data['y_et']
    eids_train = data['eids_train']; eids_iv = data['eids_iv']; eids_et = data['eids_et']

    best_params, trials_df, study, stage1_best_iters = optuna_search_xgb(
        X_train, y_train,
        imbalance_strategy=strategy,
        use_gpu=use_gpu,
        n_trials=n_trials,
        n_splits=n_folds,
        n_repeats=n_repeats,
        seed=seed,
        timeout=optuna_timeout_sec,
        search_space=search_space,
        logger_instance=logger,
    )

    trials_df.to_csv(os.path.join(strategy_checkpoint, "optuna_study.csv"), index=False)
    with open(os.path.join(strategy_checkpoint, "optuna_best_params.json"), 'w') as f:
        json.dump({
            'disease': disease_label,
            'strategy': strategy,
            'best_mean_cv_auc': float(study.best_value),
            'best_params': best_params,
            'best_iterations_at_best_trial': stage1_best_iters,
            'n_trials': n_trials,
            'n_folds': n_folds,
            'n_repeats': n_repeats,
            'seed': seed,
            'search_space': search_space,
        }, f, indent=2)

    if stage1_best_iters:
        fixed_n_estimators = max(int(np.median(stage1_best_iters)), 50)
    else:
        fixed_n_estimators = 200
    logger.info(f"[Stage 2] fixed n_estimators = {fixed_n_estimators}")

    (best_alpha_placeholder, median_model, fold_models, all_fold_indices,
     cv_results_detailed, cv_results_summary, median_fold_info) = final_cv_with_best_params(
        X_train, y_train, X_iv, y_iv, X_et, y_et,
        eids_train=eids_train, eids_internal_val=eids_iv, eids_external_test=eids_et,
        best_params=best_params,
        fixed_n_estimators=fixed_n_estimators,
        imbalance_strategy=strategy,
        use_gpu=use_gpu,
        n_folds=n_folds,
        seed=seed,
        feature_names=feature_names,
        checkpoint_dir=strategy_checkpoint,
        logger_instance=logger,
    )

    cv_results_detailed.to_csv(os.path.join(strategy_checkpoint, "cv_results_detailed.csv"), index=False)
    cv_results_summary.to_csv(os.path.join(strategy_checkpoint, "cv_results_summary.csv"), index=False)
    pd.DataFrame([median_fold_info]).to_csv(
        os.path.join(strategy_checkpoint, "median_fold_info.csv"), index=False,
    )
    with open(os.path.join(strategy_checkpoint, "median_fold_idx.txt"), 'w') as f:
        f.write(str(int(median_fold_info['median_fold_idx'])))

    importance_df = get_feature_importance_df(median_model, feature_names)
    importance_df.to_csv(os.path.join(strategy_checkpoint, "feature_importance.csv"), index=False)

    for h in logger.handlers:
        h.flush()

    return {
        'strategy': strategy,
        'cv_results_detailed': cv_results_detailed,
        'cv_results_summary': cv_results_summary,
        'median_fold_info': median_fold_info,
        'best_params': best_params,
        'fixed_n_estimators': fixed_n_estimators,
    }


def train_single_comparison(
    disease_label: str,
    train_csv: str,
    external_test_csv,
    checkpoint_dir: str,
    strategies: list,
    n_folds: int,
    n_trials: int,
    n_repeats: int,
    optuna_timeout_sec: int,
    use_gpu: bool,
    seed: int,
    search_space: dict,
    logger,
    disease_cols: list,
    filter_groups: list,
    measurement_cols: list,
    virtual_label_map: dict,
):
    logger.info("\n" + "="*80)
    logger.info(f"TRAINING: {disease_label} vs rest (strategies={strategies})")
    logger.info("="*80)

    disease_checkpoint = os.path.join(checkpoint_dir, f"{disease_label}_vs_rest")
    os.makedirs(disease_checkpoint, exist_ok=True)

    data_module = VisionECGXGBoostDataModule(
        train_csv=train_csv,
        target_disease=disease_label,
        disease_cols=disease_cols,
        filter_groups=filter_groups,
        measurement_cols=measurement_cols,
        external_test_csv=external_test_csv,
        train_ratio=0.64,
        internal_val_ratio=0.16,
        external_test_ratio=0.20,
        seed=seed,
        virtual_label_map=virtual_label_map,
    )
    X_train, y_train, X_iv, y_iv, X_et, y_et, eids_train, eids_iv, eids_et = data_module.prepare_data()
    feature_names = data_module.get_feature_names()

    logger.info(f"Data loaded: train={len(X_train)}, iv={len(X_iv)}, et={len(X_et)}, "
                f"features={X_train.shape[1]}")

    data = dict(X_train=X_train, y_train=y_train,
                X_iv=X_iv, y_iv=y_iv,
                X_et=X_et, y_et=y_et,
                eids_train=eids_train, eids_iv=eids_iv, eids_et=eids_et)

    per_strategy_results = {}
    for strategy in strategies:
        strategy_checkpoint = os.path.join(disease_checkpoint, strategy)
        os.makedirs(strategy_checkpoint, exist_ok=True)
        per_strategy_results[strategy] = train_one_strategy(
            strategy=strategy,
            disease_label=disease_label,
            data=data,
            feature_names=feature_names,
            strategy_checkpoint=strategy_checkpoint,
            n_folds=n_folds,
            n_trials=n_trials,
            n_repeats=n_repeats,
            optuna_timeout_sec=optuna_timeout_sec,
            use_gpu=use_gpu,
            seed=seed,
            search_space=search_space,
            logger=logger,
        )

    combined_rows = []
    for strategy, res in per_strategy_results.items():
        df = res['cv_results_detailed'].copy()
        df['imbalance_strategy'] = strategy
        combined_rows.append(df)
    combined_df = pd.concat(combined_rows, ignore_index=True)
    combined_df.to_csv(os.path.join(disease_checkpoint, 'cv_results_combined.csv'), index=False)

    # Cross-strategy winner = strategy with highest MEDIAN internal_val_auc.
    strategy_median_iv = (
        combined_df.groupby('imbalance_strategy')['internal_val_auc']
        .median()
        .sort_values(ascending=False)
    )
    winning_strategy = str(strategy_median_iv.index[0])
    logger.info(f"\nCross-strategy median internal_val_auc:\n{strategy_median_iv.to_string()}")
    logger.info(f"Winning strategy: {winning_strategy}")

    winning_median_info = per_strategy_results[winning_strategy]['median_fold_info']
    winning_row = dict(winning_median_info)
    winning_row['winning_strategy'] = winning_strategy
    pd.DataFrame([winning_row]).to_csv(
        os.path.join(disease_checkpoint, 'winning_strategy_median_fold_info.csv'), index=False,
    )

    median_row = {'disease': disease_label}
    for col in combined_df.columns:
        if col in ('fold', 'imbalance_strategy', 'alpha', 'best_iteration'):
            continue
        try:
            median_row[col] = float(combined_df[col].median())
        except (TypeError, ValueError):
            pass
    pd.DataFrame([median_row]).to_csv(
        os.path.join(disease_checkpoint, 'median_across_all_info.csv'), index=False,
    )

    for h in logger.handlers:
        h.flush()

    return {
        'disease': disease_label,
        'checkpoint_dir': disease_checkpoint,
        'combined_cv_path': os.path.join(disease_checkpoint, 'cv_results_combined.csv'),
        'winning_strategy': winning_strategy,
        'winning_median_fold_path': os.path.join(disease_checkpoint, 'winning_strategy_median_fold_info.csv'),
        'median_across_all_path': os.path.join(disease_checkpoint, 'median_across_all_info.csv'),
    }


def aggregate_performance_summary(all_results, output_path, logger):
    """Median [min,max] across every (strategy,fold) per disease."""
    logger.info("\n" + "="*80)
    logger.info("AGGREGATING MEDIAN PERFORMANCE SUMMARY")
    logger.info("="*80)

    summary_rows = []
    for result in all_results:
        disease = result['disease']
        combined_path = result['combined_cv_path']
        if not os.path.exists(combined_path):
            logger.warning(f"  Missing combined CSV: {combined_path}")
            continue

        combined_df = pd.read_csv(combined_path)
        row = {'comparison': f"{disease}_vs_rest"}
        for dataset in ['train', 'internal_val', 'external_val']:
            for metric in ['auc', 'auprc', 'tpr', 'tnr', 'fpr', 'fnr', 'f1']:
                col = f"{dataset}_{metric}"
                if col in combined_df.columns:
                    values = combined_df[col].dropna().values
                    if len(values):
                        median = float(np.median(values))
                        min_val = float(np.min(values))
                        max_val = float(np.max(values))
                        row[col] = f"{median:.4f} [{min_val:.4f}, {max_val:.4f}]"
                    else:
                        row[col] = "N/A"
                else:
                    row[col] = "N/A"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path}")
    for h in logger.handlers:
        h.flush()
    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="visionECG-XGBoost multi-label one-vs-rest classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--external_test_csv", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, default="visionecg_xgboost_")
    parser.add_argument(
        "--no_dt_suffix", action='store_true',
        help="Use --checkpoint_dir verbatim (skip appending output_prefix + timestamp).",
    )

    parser.add_argument("--diseases", nargs='+', type=str, required=True,
                        help="Target diseases for one-vs-rest classification.")
    parser.add_argument("--disease_cols", nargs='+', type=str, required=True,
                        help="Full list of disease label columns present in the CSV.")
    parser.add_argument("--filter_groups", nargs='+', type=str, required=True,
                        help="Cohort filter (rows must belong to at least one of these).")
    parser.add_argument("--measurement_cols", nargs='+', type=str, required=True,
                        help="Feature columns (visionECG phenotypes).")
    parser.add_argument(
        "--virtual_label_map", type=str, default=None,
        help="Optional python-literal dict mapping virtual label -> [physical cols], "
             "e.g. \"{'CM': ['HCM', 'DCM']}\"",
    )

    parser.add_argument("--strategies", nargs='+', type=str, default=['spw_only'],
                        choices=['spw_only','none'],
                        help="Class-imbalance strategies (default: spw_only).")
    parser.add_argument("--search_space_json", type=str, required=True,
                        help="JSON string mapping param name -> {type,low,high,[log]}. "
                             "Fully defines the Optuna search space; nothing hardcoded in Python.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--n_trials", type=int, required=True)
    parser.add_argument("--optuna_timeout_sec", type=int, default=3600)

    parser.add_argument("--use_gpu", action='store_true')

    args = parser.parse_args()

    if isinstance(args.external_test_csv, str) and args.external_test_csv.strip().lower() in ('', 'none', 'null'):
        args.external_test_csv = None

    virtual_label_map = _parse_virtual_label_map(args.virtual_label_map)

    try:
        search_space = json.loads(args.search_space_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--search_space_json is not valid JSON: {exc}")
    if not isinstance(search_space, dict) or not search_space:
        raise SystemExit("--search_space_json must be a non-empty JSON object.")
    for name, spec in search_space.items():
        if not isinstance(spec, dict) or 'type' not in spec:
            raise SystemExit(f"--search_space_json['{name}'] must be an object with a 'type' field.")

    for disease in args.diseases:
        if disease not in args.filter_groups and disease not in virtual_label_map:
            raise ValueError(
                f"disease '{disease}' is not in filter_groups {args.filter_groups} "
                f"and not in virtual_label_map keys {list(virtual_label_map)}."
            )

    if args.no_dt_suffix:
        checkpoint_dir = args.checkpoint_dir
    else:
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = args.checkpoint_dir + args.output_prefix + dt_string
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = setup_logging(checkpoint_dir)
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info(f"Train CSV: {args.train_csv}")
    logger.info(f"External test CSV: {args.external_test_csv}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info(f"Diseases ({len(args.diseases)}): {args.diseases}")
    logger.info(f"Filter groups: {args.filter_groups}")
    logger.info(f"Disease cols ({len(args.disease_cols)}): {args.disease_cols}")
    logger.info(f"Measurement cols ({len(args.measurement_cols)}): {args.measurement_cols}")
    logger.info(f"Virtual label map: {virtual_label_map}")
    logger.info(f"Strategies: {args.strategies}")
    logger.info(f"Search space ({len(search_space)} params): {sorted(search_space.keys())}")
    logger.info(f"Seed: {args.seed}, folds: {args.n_folds}, repeats: {args.n_repeats}, "
                f"trials: {args.n_trials}, timeout: {args.optuna_timeout_sec}s")
    logger.info(f"Use GPU: {args.use_gpu}")
    logger.info(f"XGB_N_JOBS: {os.environ.get('XGB_N_JOBS', '1')}")
    for h in logger.handlers:
        h.flush()

    all_results = []
    for i, disease in enumerate(args.diseases, 1):
        logger.info(f"\n{'='*80}\nComparison {i}/{len(args.diseases)}: {disease} vs rest\n{'='*80}")
        try:
            result = train_single_comparison(
                disease_label=disease,
                train_csv=args.train_csv,
                external_test_csv=args.external_test_csv,
                checkpoint_dir=checkpoint_dir,
                strategies=args.strategies,
                n_folds=args.n_folds,
                n_trials=args.n_trials,
                n_repeats=args.n_repeats,
                optuna_timeout_sec=args.optuna_timeout_sec,
                use_gpu=args.use_gpu,
                seed=args.seed,
                search_space=search_space,
                logger=logger,
                disease_cols=args.disease_cols,
                filter_groups=args.filter_groups,
                measurement_cols=args.measurement_cols,
                virtual_label_map=virtual_label_map,
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error training {disease} vs rest: {e}", exc_info=True)
            logger.error("Continuing with remaining comparisons...")

    if all_results:
        summary_path = os.path.join(checkpoint_dir, "multilabel_performance_summary.csv")
        aggregate_performance_summary(all_results, summary_path, logger)
    else:
        logger.error("No successful comparisons to aggregate")

    logger.info("\n" + "="*80)
    logger.info("visionECG-XGBoost CLASSIFICATION COMPLETED")
    logger.info("="*80)
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info(f"Completed diseases ({len(all_results)}/{len(args.diseases)}):")
    for result in all_results:
        logger.info(f"  {result['disease']} vs rest: winner={result['winning_strategy']} "
                    f"-> {result['checkpoint_dir']}")
    logger.info(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for h in logger.handlers:
        h.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
