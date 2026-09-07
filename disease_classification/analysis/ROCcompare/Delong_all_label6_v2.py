#!/usr/bin/env python3
"""Batch DeLong, NRI, and IDI across disease models."""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from itertools import combinations

import pandas as pd


def setup_logging(output_dir: str) -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"delong_run_{timestamp}.log")

    logger = logging.getLogger("delong_all_label6_v2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")
    return logger


def slugify(name: str) -> str:
    """Whitespace/special -> underscore."""
    return name.replace(' ', '_').replace('&', 'and')


def build_config_from_tables(input_tables, model_names, prob_cols, diseases):
    """Build MODEL_CONFIG from per-model paths tables."""
    if not (len(input_tables) == len(model_names) == len(prob_cols)):
        raise ValueError(
            f"input_tables/model_names/prob_cols must have equal length; "
            f"got {len(input_tables)}, {len(model_names)}, {len(prob_cols)}"
        )
    config = {}
    for table_path, name, prob_col in zip(input_tables, model_names, prob_cols):
        df = pd.read_csv(table_path)
        if 'dataset_label' not in df.columns or 'test_path' not in df.columns:
            raise ValueError(
                f"Table {table_path} must have 'dataset_label' and 'test_path'; "
                f"got {list(df.columns)}"
            )
        label_to_path = {}
        for _, row in df.iterrows():
            canon = str(row['dataset_label']).strip()
            if canon not in diseases:
                raise ValueError(
                    f"{name}: unknown dataset_label {canon!r}; expected one of {diseases}"
                )
            if canon in label_to_path:
                raise ValueError(f"{name}: duplicate rows for {canon!r}")
            label_to_path[canon] = str(row['test_path'])
        files = []
        for d in diseases:
            p = label_to_path.get(d)
            files.append(p if (isinstance(p, str) and p.strip()) else None)
        config[name] = {
            'short_name': name,
            'prob_col': prob_col,
            'files': files,
        }
    return config


def resolve_path(relative_path: str, base_dir: str) -> str:
    return os.path.join(base_dir, relative_path)


def validate_file_exists(file_path: str, logger: logging.Logger) -> bool:
    if os.path.exists(file_path):
        return True
    logger.warning(f"File not found: {file_path}")
    return False


def generate_model_pairs(model_config: dict):
    model_names = list(model_config.keys())
    pairs = []
    for m1, m2 in combinations(model_names, 2):
        short1 = model_config[m1]['short_name']
        short2 = model_config[m2]['short_name']
        pair_name = f"{slugify(short1)}_vs_{slugify(short2)}"
        pairs.append({
            'model1_name': m1,
            'model2_name': m2,
            'model1_short': short1,
            'model2_short': short2,
            'pair_name': pair_name,
        })
    return pairs


def run_single_delong_test(
    csv1_path, csv2_path, model1_name, model2_name, disease_name,
    output_dir, delong_script, rscript_path, prob_col1, prob_col2, logger,
) -> pd.DataFrame:
    logger.info(f"  Running DeLong test: {model1_name} vs {model2_name} for {disease_name}")

    cmd = [
        rscript_path,
        delong_script,
        "--csv1", csv1_path,
        "--csv2", csv2_path,
        "--model1_name", model1_name,
        "--model2_name", model2_name,
        "--task_name", disease_name,
        "--output", output_dir,
        "--prob_col1", prob_col1,
        "--prob_col2", prob_col2,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error(f"  R script failed: {result.stderr}")
            return None

        result_file = os.path.join(output_dir, f"{disease_name}_delong_results.csv")
        if os.path.exists(result_file):
            df = pd.read_csv(result_file)
            logger.info(f"  Results saved to: {result_file}")
            return df
        logger.error(f"  Result file not found: {result_file}")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"  Timeout running DeLong test for {disease_name}")
        return None
    except Exception as e:
        logger.error(f"  Error running DeLong test: {e}")
        return None


def process_all_comparisons(
    model_config: dict,
    output_base_dir: str,
    delong_script: str,
    rscript_path: str,
    base_dir: str,
    diseases: list,
    logger: logging.Logger,
) -> pd.DataFrame:
    pairs = generate_model_pairs(model_config)
    logger.info(f"Generated {len(pairs)} model pairs for comparison")

    all_results = []
    warnings_log = []

    for pair in pairs:
        pair_name = pair['pair_name']
        model1_name = pair['model1_name']
        model2_name = pair['model2_name']
        prob_col1 = model_config[model1_name].get('prob_col', 'predicted_probability')
        prob_col2 = model_config[model2_name].get('prob_col', 'predicted_probability')

        logger.info(f"\n{'='*80}")
        logger.info(f"Processing pair: {pair_name}")
        logger.info(f"  {model1_name} vs {model2_name}")
        logger.info(f"  prob cols: {prob_col1} / {prob_col2}")
        logger.info(f"{'='*80}")

        pair_output_dir = os.path.join(output_base_dir, pair_name)
        os.makedirs(pair_output_dir, exist_ok=True)

        for disease_idx, disease_name in enumerate(diseases):
            logger.info(f"\n  Disease {disease_idx + 1}/{len(diseases)}: {disease_name}")

            csv1_relative = model_config[model1_name]['files'][disease_idx]
            csv2_relative = model_config[model2_name]['files'][disease_idx]

            if csv1_relative is None:
                msg = f"skipped: {model1_name} has no file for {disease_name}"
                logger.info(f"  {msg}")
                warnings_log.append({'pair': pair_name, 'disease': disease_name,
                                     'model': model1_name, 'error': msg})
                continue
            if csv2_relative is None:
                msg = f"skipped: {model2_name} has no file for {disease_name}"
                logger.info(f"  {msg}")
                warnings_log.append({'pair': pair_name, 'disease': disease_name,
                                     'model': model2_name, 'error': msg})
                continue

            csv1_path = resolve_path(csv1_relative, base_dir)
            csv2_path = resolve_path(csv2_relative, base_dir)

            if not validate_file_exists(csv1_path, logger):
                warnings_log.append({'pair': pair_name, 'disease': disease_name,
                                     'model': model1_name, 'error': f"File not found: {csv1_path}"})
                continue
            if not validate_file_exists(csv2_path, logger):
                warnings_log.append({'pair': pair_name, 'disease': disease_name,
                                     'model': model2_name, 'error': f"File not found: {csv2_path}"})
                continue

            result_df = run_single_delong_test(
                csv1_path=csv1_path,
                csv2_path=csv2_path,
                model1_name=model1_name,
                model2_name=model2_name,
                disease_name=disease_name,
                output_dir=pair_output_dir,
                delong_script=delong_script,
                rscript_path=rscript_path,
                prob_col1=prob_col1,
                prob_col2=prob_col2,
                logger=logger,
            )

            if result_df is not None:
                result_df['model_pair'] = pair_name
                all_results.append(result_df)
            else:
                warnings_log.append({'pair': pair_name, 'disease': disease_name,
                                     'model': 'N/A', 'error': 'DeLong test failed'})

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        if 'task' in combined_df.columns and 'disease' not in combined_df.columns:
            combined_df = combined_df.rename(columns={'task': 'disease'})

        col_order = ['model_pair', 'disease', 'model1_name', 'model2_name',
                     'n_patients', 'n_positive', 'n_negative',
                     'model1_auc', 'model1_auc_ci_lower', 'model1_auc_ci_upper',
                     'model2_auc', 'model2_auc_ci_lower', 'model2_auc_ci_upper',
                     'auc_difference', 'auc_diff_ci_lower', 'auc_diff_ci_upper',
                     'delong_p_value_two_sided', 'delong_p_value_one_sided',
                     'continuous_nri', 'continuous_nri_se',
                     'continuous_nri_ci_lower', 'continuous_nri_ci_upper',
                     'continuous_nri_p_value',
                     'idi', 'idi_se', 'idi_ci_lower', 'idi_ci_upper', 'idi_p_value']
        col_order = [c for c in col_order if c in combined_df.columns]
        combined_df = combined_df[col_order]

        output_path = os.path.join(output_base_dir, "all_delong_results.csv")
        combined_df.to_csv(output_path, index=False)
        logger.info(f"\nAggregated results saved to: {output_path} ({len(combined_df)} rows)")

        sig_count = (combined_df['delong_p_value_two_sided'] < 0.05).sum()
        logger.info(f"Significant differences (p < 0.05): {sig_count}/{len(combined_df)}")
    else:
        logger.error("No results to aggregate!")
        combined_df = pd.DataFrame()

    if warnings_log:
        warnings_df = pd.DataFrame(warnings_log)
        warnings_path = os.path.join(output_base_dir, "delong_warnings.csv")
        warnings_df.to_csv(warnings_path, index=False)
        logger.warning(f"Warnings saved to: {warnings_path} ({len(warnings_log)})")

    return combined_df


def main():
    parser = argparse.ArgumentParser(
        description="Run DeLong tests comparing N label6 disease-prediction models across 6 diseases."
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_dir", type=str, default='',
                        help="Base directory for resolving relative paths.")
    parser.add_argument("--delong_script", type=str, required=True,
                        help="Path to Delong_measurements_v2.R (absolute or relative to --base_dir).")
    parser.add_argument("--rscript_path", type=str, required=True,
                        help="Path to Rscript executable.")
    parser.add_argument("--input_tables", nargs='+', type=str, required=True,
                        help="Per-model paths tables (columns: dataset_label, test_path).")
    parser.add_argument("--model_names", nargs='+', type=str, required=True,
                        help="Display names aligned with --input_tables.")
    parser.add_argument("--prob_cols", nargs='+', type=str, required=True,
                        help="Probability column names aligned with --input_tables.")
    parser.add_argument("--diseases", nargs='+', type=str, required=True,
                        help="Canonical disease labels; defines validation set and iteration order.")

    args = parser.parse_args()

    model_config = build_config_from_tables(
        args.input_tables, args.model_names, args.prob_cols, args.diseases,
    )

    output_dir = resolve_path(args.output_dir, args.base_dir)
    os.makedirs(output_dir, exist_ok=True)

    delong_script = args.delong_script if os.path.isabs(args.delong_script) \
        else resolve_path(args.delong_script, args.base_dir)

    logger = setup_logging(output_dir)

    logger.info(f"Output: {output_dir} | base_dir={args.base_dir!r}")
    logger.info(f"DeLong R script: {delong_script} | Rscript: {args.rscript_path}")
    logger.info(f"Models: {list(model_config.keys())}")
    logger.info(f"Diseases: {args.diseases}")
    n_pairs = len(list(combinations(model_config.keys(), 2)))
    logger.info(f"Cells: {n_pairs} pairs x {len(args.diseases)} diseases = {n_pairs * len(args.diseases)}")

    if not os.path.exists(delong_script):
        logger.error(f"DeLong R script not found: {delong_script}")
        sys.exit(1)

    start_time = datetime.now()

    results_df = process_all_comparisons(
        model_config=model_config,
        output_base_dir=output_dir,
        delong_script=delong_script,
        rscript_path=args.rscript_path,
        base_dir=args.base_dir,
        diseases=args.diseases,
        logger=logger,
    )

    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"Done in {duration:.2f}s. Results dir: {output_dir}")

    if not results_df.empty:
        for pair in results_df['model_pair'].unique():
            pair_df = results_df[results_df['model_pair'] == pair]
            sig_count = (pair_df['delong_p_value_two_sided'] < 0.05).sum()
            logger.info(f"  {pair}: {sig_count}/{len(pair_df)} significant")


if __name__ == "__main__":
    main()
