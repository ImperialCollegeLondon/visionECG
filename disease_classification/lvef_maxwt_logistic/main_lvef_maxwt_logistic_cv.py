#!/usr/bin/env python3
"""LVEF + max-WT one-vs-rest logistic CV wrapper."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import sklearn


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from loader_lvef_maxwt_logistic import LvefMaxwtLogisticDataModule  
from model_lvef_maxwt_logistic import run_logistic_cv  


SUMMARY_DATASETS = ("train", "internal_val", "external_test")
SUMMARY_METRICS = ("auc", "auprc", "tpr", "tnr", "fpr", "fnr", "f1")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected true or false, received {value!r}."
    )


def parse_virtual_label_map(raw: Optional[str]) -> Dict[str, List[str]]:
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
    return {str(k): list(v) for k, v in parsed.items()}


def setup_logging(run_dir: Path) -> logging.Logger:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "lvef_maxwt_logistic_wrapper.log"

    logger = logging.getLogger("lvef_maxwt_logistic")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info("Log file: %s", log_path)
    return logger


def build_run_dir(
    checkpoint_dir: str, output_prefix: str, no_dt_suffix: bool
) -> Path:
    checkpoint_base = Path(checkpoint_dir)
    if no_dt_suffix:
        run_dir = checkpoint_base
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = checkpoint_base / f"{output_prefix}{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_config(
    args: argparse.Namespace,
    run_dir: Path,
    resolved_diseases: List[str],
    virtual_label_map: Dict[str, List[str]],
) -> Path:
    config: Dict[str, Any] = {
        "command": sys.argv,
        "created_at": datetime.now().astimezone().isoformat(),
        "run_dir": str(run_dir),
        "arguments": vars(args),
        "resolved_diseases": list(resolved_diseases),
        "disease_cols": list(args.disease_cols),
        "filter_groups": list(args.filter_groups),
        "feature_cols": list(args.feature_cols),
        "virtual_label_map": virtual_label_map,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    output_path = run_dir / "run_config.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    return output_path


def _format_median_range(values: Iterable[float]) -> str:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if numeric.empty:
        return "N/A"
    return (
        f"{numeric.median():.4f} "
        f"[{numeric.min():.4f}, {numeric.max():.4f}]"
    )


def aggregate_median_summary(
    results: List[Dict[str, Any]], output_path: Path, logger: logging.Logger
) -> pd.DataFrame:
    """Aggregate every fold as median [minimum, maximum]."""
    rows: List[Dict[str, Any]] = []
    for result in results:
        detailed = pd.read_csv(result["cv_detailed"])
        row: Dict[str, Any] = {"comparison": f"{result['disease']}_vs_rest"}
        for dataset in SUMMARY_DATASETS:
            for metric in SUMMARY_METRICS:
                column = f"{dataset}_{metric}"
                row[column] = (
                    _format_median_range(detailed[column])
                    if column in detailed.columns
                    else "N/A"
                )
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False)
    logger.info("Saved median fold summary: %s", output_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LVEF + max-WT one-vs-rest logistic regression with stratified CV."
    )
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--external_test_csv", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_prefix", default="lvef_maxwt_logistic_")
    parser.add_argument(
        "--no_dt_suffix",
        action="store_true",
        help="Use --checkpoint_dir verbatim instead of adding a timestamped child.",
    )
    parser.add_argument("--diseases", nargs="+", required=True)
    parser.add_argument("--disease_cols", nargs="+", required=True)
    parser.add_argument("--filter_groups", nargs="+", required=True)
    parser.add_argument("--feature_cols", nargs="+", required=True)
    parser.add_argument(
        "--virtual_label_map",
        type=str,
        default=None,
        help="Python-literal dict mapping virtual label -> [physical cols], "
             "e.g. \"{'CM': ['HCM', 'DCM']}\".",
    )
    parser.add_argument("--n_folds", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--selection_metric", choices=("auc", "loss"), required=True)
    parser.add_argument(
        "--use_pos_weight",
        type=parse_bool,
        required=True,
        metavar="{true,false}",
        help="If true, weight positive fold-training rows by n_negative/n_positive.",
    )
    parser.add_argument("--eid_col", default="eid_18545")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    virtual_label_map = parse_virtual_label_map(args.virtual_label_map)

    supported_targets = list(dict.fromkeys([*args.filter_groups, *args.disease_cols]))
    for disease in args.diseases:
        if disease not in supported_targets and disease not in virtual_label_map:
            raise SystemExit(
                f"Invalid disease target '{disease}'; valid targets are {supported_targets} "
                f"or virtual labels {list(virtual_label_map)}."
            )
    if len(set(args.diseases)) != len(args.diseases):
        raise SystemExit(f"Duplicate disease targets are not allowed: {args.diseases}")
    if args.n_folds < 2:
        raise SystemExit("--n_folds must be at least 2.")

    run_dir = build_run_dir(
        args.checkpoint_dir, args.output_prefix, args.no_dt_suffix
    )
    logger = setup_logging(run_dir)
    config_path = save_run_config(args, run_dir, args.diseases, virtual_label_map)

    logger.info("=" * 80)
    logger.info("LVEF + MAX-WT ONE-VS-REST LOGISTIC REGRESSION")
    logger.info("=" * 80)
    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Train CSV: %s", args.train_csv)
    logger.info("External test CSV: %s", args.external_test_csv)
    logger.info("Run directory: %s", run_dir)
    logger.info("Run configuration: %s", config_path)
    logger.info("Targets: %s", args.diseases)
    logger.info("Filter groups: %s", args.filter_groups)
    logger.info("Disease cols (%d): %s", len(args.disease_cols), args.disease_cols)
    logger.info("Feature cols: %s", args.feature_cols)
    logger.info("Virtual label map: %s", virtual_label_map)
    logger.info("CV folds: %d; seed: %d", args.n_folds, args.seed)
    logger.info("Median-fold selection: internal_val %s", args.selection_metric)
    logger.info("Positive weighting: %s", args.use_pos_weight)
    if args.use_pos_weight:
        logger.warning(
            "Positive weighting changes the fitted class prior. The saved "
            "predicted_probability values are not prevalence-calibrated."
        )

    data_module = LvefMaxwtLogisticDataModule(
        train_csv=args.train_csv,
        external_test_csv=args.external_test_csv,
        disease_cols=args.disease_cols,
        filter_groups=args.filter_groups,
        feature_cols=args.feature_cols,
        virtual_label_map=virtual_label_map,
        eid_col=args.eid_col,
        logger=logger,
    )
    train_df, test_df, cohort_summary = data_module.prepare_cohorts()
    cohort_summary_path = run_dir / "cohort_filter_summary.csv"
    cohort_summary.to_csv(cohort_summary_path, index=False)
    logger.info("Saved cohort filter summary: %s", cohort_summary_path)

    feature_names = data_module.get_feature_names()
    X_train_raw = train_df[feature_names].to_numpy(dtype=np.float64)
    X_test_raw = test_df[feature_names].to_numpy(dtype=np.float64)
    eids_train = train_df[data_module.eid_col].to_numpy()
    eids_test = test_df[data_module.eid_col].to_numpy()

    successful: List[Dict[str, Any]] = []
    failed: List[str] = []
    for index, disease in enumerate(args.diseases, start=1):
        logger.info("\n%s", "=" * 80)
        logger.info("TASK %d/%d: %s vs rest", index, len(args.diseases), disease)
        logger.info("%s", "=" * 80)
        disease_dir = run_dir / f"{disease}_vs_rest"

        try:
            y_train = data_module.create_binary_labels(
                train_df, disease, dataset_name="train"
            )
            y_test = data_module.create_binary_labels(
                test_df,
                disease,
                dataset_name="external_test",
                require_both_classes=False,
            )
            result = run_logistic_cv(
                X_train_raw=X_train_raw,
                y_train=y_train,
                eids_train=eids_train,
                X_test_raw=X_test_raw,
                y_test=y_test,
                eids_test=eids_test,
                feature_names=feature_names,
                output_dir=disease_dir,
                n_folds=args.n_folds,
                seed=args.seed,
                selection_metric=args.selection_metric,
                use_pos_weight=args.use_pos_weight,
                logger=logger,
            )
            result.update(
                {
                    "disease": disease,
                    "selection_metric": args.selection_metric,
                    "use_pos_weight": args.use_pos_weight,
                }
            )
            successful.append(result)
        except Exception:
            failed.append(disease)
            logger.exception("Task failed for %s vs rest", disease)

    if successful:
        aggregate_median_summary(
            successful, run_dir / "multilabel_performance_summary.csv", logger
        )

    logger.info("\n%s", "=" * 80)
    logger.info("RUN COMPLETE: %d/%d tasks succeeded", len(successful), len(args.diseases))
    logger.info("Output: %s", run_dir)
    if failed:
        logger.error("Failed tasks: %s", failed)
    logger.info("%s", "=" * 80)

    for handler in logger.handlers:
        handler.flush()
    return 1 if failed or not successful else 0


if __name__ == "__main__":
    sys.exit(main())
