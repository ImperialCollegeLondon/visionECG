#!/usr/bin/env python3
"""visionECG phenotype threshold classification — 5-fold CV logistic regression."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, f1_score, confusion_matrix
)


class VisionECGPhenotypeAnalyzer:
    def __init__(self, original_path: str, compared_path: str,
                 original_id_col: str, compared_id_col: str,
                 output_dir: str = "prediction_analysis_results",
                 n_bootstrap: int = 1000,
                 cv_folds: int = 5,
                 random_state: int = 42,
                 instance_col: str = None,
                 test_original_path: str = None,
                 test_compared_path: str = None):
        """Initialize the Prediction Analyzer."""
        self.original_path = original_path
        self.compared_path = compared_path
        self.original_id_col = original_id_col
        self.compared_id_col = compared_id_col
        self.instance_col = instance_col
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.n_bootstrap = n_bootstrap
        self.cv_folds = cv_folds
        self.random_state = random_state

        self.test_original_path = test_original_path
        self.test_compared_path = test_compared_path
        self.has_test_set = test_compared_path is not None
        self.test_original_df = None
        self.test_compared_df = None

        # Setup logging
        self.setup_logging()

        # Load data
        self.logger.info("=" * 80)
        self.logger.info("PREDICTION ANALYSIS STARTED")
        self.logger.info("=" * 80)
        self.load_data()

        if self.has_test_set:
            self.load_test_data()

    def setup_logging(self):
        """Setup comprehensive logging system."""
        # Create logger
        self.logger = logging.getLogger('VisionECGPhenotypeAnalyzer')
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers = []

        # Create console handler with formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(console_handler)

        # Also create file logger
        log_file = self.output_dir / f"prediction_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.logger.info(f"Log file created: {log_file}")

    def load_data(self):
        """Load and validate data files."""
        self.logger.info(f"Loading original data from: {self.original_path}")
        try:
            self.original_df = pd.read_csv(self.original_path)
            self.logger.info(f"Original data loaded: {self.original_df.shape[0]} rows, {self.original_df.shape[1]} columns")
            self.logger.info(f"Original ID column: '{self.original_id_col}'")

            if self.original_id_col not in self.original_df.columns:
                self.logger.error(f"ID column '{self.original_id_col}' not found in original data")
                self.logger.info(f"Available columns: {list(self.original_df.columns)}")
                raise ValueError(f"ID column '{self.original_id_col}' not found")

        except Exception as e:
            self.logger.error(f"Failed to load original data: {e}")
            raise

        self.logger.info(f"Loading compared data from: {self.compared_path}")
        try:
            self.compared_df = pd.read_csv(self.compared_path)
            self.logger.info(f"Compared data loaded: {self.compared_df.shape[0]} rows, {self.compared_df.shape[1]} columns")
            self.logger.info(f"Compared ID column: '{self.compared_id_col}'")

            if self.compared_id_col not in self.compared_df.columns:
                self.logger.error(f"ID column '{self.compared_id_col}' not found in compared data")
                self.logger.info(f"Available columns: {list(self.compared_df.columns)}")
                raise ValueError(f"ID column '{self.compared_id_col}' not found")

        except Exception as e:
            self.logger.error(f"Failed to load compared data: {e}")
            raise

    def load_test_data(self):
        """Load held-out test CSVs."""
        self.logger.info(f"Loading TEST compared data from: {self.test_compared_path}")
        try:
            self.test_compared_df = pd.read_csv(self.test_compared_path)
            self.logger.info(f"Test compared loaded: {self.test_compared_df.shape[0]} rows, {self.test_compared_df.shape[1]} columns")
            if self.compared_id_col not in self.test_compared_df.columns:
                self.logger.error(f"ID column '{self.compared_id_col}' not found in test compared data")
                raise ValueError(f"ID column '{self.compared_id_col}' not found in test compared")
        except Exception as e:
            self.logger.error(f"Failed to load test compared data: {e}")
            raise

        if self.test_original_path is None or self.test_original_path == self.original_path:
            self.logger.info("Reusing training ground-truth (--original_path) for the test merge")
            self.test_original_df = self.original_df
        else:
            self.logger.info(f"Loading TEST original data from: {self.test_original_path}")
            try:
                self.test_original_df = pd.read_csv(self.test_original_path)
                self.logger.info(f"Test original loaded: {self.test_original_df.shape[0]} rows, {self.test_original_df.shape[1]} columns")
                if self.original_id_col not in self.test_original_df.columns:
                    raise ValueError(f"ID column '{self.original_id_col}' not found in test original")
            except Exception as e:
                self.logger.error(f"Failed to load test original data: {e}")
                raise

    def analyze_id_overlap_test(self) -> pd.DataFrame:
        """Merge held-out truth and predictions by patient ID."""
        self.logger.info("-" * 40)
        self.logger.info("TEST ID OVERLAP ANALYSIS")
        self.logger.info("-" * 40)

        original_ids = set(self.test_original_df[self.original_id_col].dropna())
        compared_ids = set(self.test_compared_df[self.compared_id_col].dropna())
        overlapping_ids = original_ids.intersection(compared_ids)
        self.logger.info(f"Test original unique IDs: {len(original_ids)}")
        self.logger.info(f"Test compared unique IDs: {len(compared_ids)}")
        self.logger.info(f"Test overlapping IDs: {len(overlapping_ids)}")
        if len(overlapping_ids) == 0:
            raise ValueError("No overlapping IDs found in test set")

        original_filtered = self.test_original_df[
            self.test_original_df[self.original_id_col].isin(overlapping_ids)
        ].copy()
        compared_filtered = self.test_compared_df[
            self.test_compared_df[self.compared_id_col].isin(overlapping_ids)
        ].copy()

        merged_df = original_filtered.merge(
            compared_filtered,
            left_on=self.original_id_col,
            right_on=self.compared_id_col,
            how='inner',
            suffixes=('_original', '_compared')
        )
        self.logger.info(f"Test merged data: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
        return merged_df

    def analyze_id_overlap(self) -> pd.DataFrame:
        """Analyze ID overlap between tables and merge data."""
        self.logger.info("-" * 40)
        self.logger.info("ID OVERLAP ANALYSIS")
        self.logger.info("-" * 40)

        # Get unique IDs from each table
        original_ids = set(self.original_df[self.original_id_col].dropna())
        compared_ids = set(self.compared_df[self.compared_id_col].dropna())

        self.logger.info(f"Original table unique IDs: {len(original_ids)}")
        self.logger.info(f"Compared table unique IDs: {len(compared_ids)}")

        # Find overlapping IDs
        overlapping_ids = original_ids.intersection(compared_ids)

        self.logger.info(f"Overlapping IDs: {len(overlapping_ids)}")
        self.logger.info(f"IDs only in original: {len(original_ids - compared_ids)}")
        self.logger.info(f"IDs only in compared: {len(compared_ids - original_ids)}")

        if len(overlapping_ids) == 0:
            self.logger.error("No overlapping IDs found between tables!")
            raise ValueError("No overlapping IDs found")

        # Filter data to overlapping IDs only
        original_filtered = self.original_df[
            self.original_df[self.original_id_col].isin(overlapping_ids)
        ].copy()
        compared_filtered = self.compared_df[
            self.compared_df[self.compared_id_col].isin(overlapping_ids)
        ].copy()

        self.logger.info(f"Original data after ID filtering: {original_filtered.shape[0]} rows")
        self.logger.info(f"Compared data after ID filtering: {compared_filtered.shape[0]} rows")

        # Merge data
        merged_df = original_filtered.merge(
            compared_filtered,
            left_on=self.original_id_col,
            right_on=self.compared_id_col,
            how='inner',
            suffixes=('_original', '_compared')
        )

        self.logger.info(f"Final merged data: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")

        return merged_df

    def filter_available_columns(self, column_mapping: Dict[str, List[str]],
                                merged_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Retain mappings for available columns."""
        self.logger.info("-" * 40)
        self.logger.info("FILTERING AVAILABLE COLUMNS")
        self.logger.info("-" * 40)

        variables = column_mapping['Variables']
        original_cols = column_mapping['original_cols']
        compared_cols = column_mapping['compared_cols']
        thresholds = column_mapping.get('thresholds', [])
        threshold_directions = column_mapping.get('threshold_directions', [])

        filtered_variables = []
        filtered_original_cols = []
        filtered_compared_cols = []
        filtered_thresholds = []
        filtered_threshold_directions = []

        for idx, (var_name, orig_col, comp_col) in enumerate(zip(variables, original_cols, compared_cols)):
            orig_in_source = orig_col in self.original_df.columns
            comp_in_source = comp_col in self.compared_df.columns

            if not orig_in_source or not comp_in_source:
                missing = []
                if not orig_in_source:
                    missing.append(f"original column '{orig_col}' (not found in original table)")
                if not comp_in_source:
                    missing.append(f"compared column '{comp_col}' (not found in compared table)")
                self.logger.warning(f"✗ {var_name}: Missing {', '.join(missing)}")
                continue


            if orig_col in self.compared_df.columns:
                # Shared columns receive the _original suffix.
                orig_col_suffixed = f"{orig_col}_original"
            else:
                # Original-only columns keep their names.
                orig_col_suffixed = orig_col

            if comp_col in self.original_df.columns:
                # Shared columns receive the _compared suffix.
                comp_col_suffixed = f"{comp_col}_compared"
            else:
                # Compared-only columns keep their names.
                comp_col_suffixed = comp_col

            # Verify resolved columns exist after merging.
            orig_exists = orig_col_suffixed in merged_df.columns
            comp_exists = comp_col_suffixed in merged_df.columns

            if orig_exists and comp_exists:
                filtered_variables.append(var_name)
                filtered_original_cols.append(orig_col_suffixed)
                filtered_compared_cols.append(comp_col_suffixed)

                # Add threshold and direction if provided
                if idx < len(thresholds):
                    filtered_thresholds.append(thresholds[idx])
                if idx < len(threshold_directions):
                    filtered_threshold_directions.append(threshold_directions[idx])

                self.logger.info(f"✓ {var_name}: '{orig_col_suffixed}' <-> '{comp_col_suffixed}'")
            else:
                # Log unexpected missing merged columns.
                missing = []
                if not orig_exists:
                    missing.append(f"'{orig_col_suffixed}' in merged dataframe")
                if not comp_exists:
                    missing.append(f"'{comp_col_suffixed}' in merged dataframe")
                self.logger.error(f"✗ {var_name}: Unexpected error - columns passed source validation "
                                f"but missing from merged dataframe: {', '.join(missing)}")

        filtered_mapping = {
            'Variables': filtered_variables,
            'original_cols': filtered_original_cols,
            'compared_cols': filtered_compared_cols
        }

        # Only add thresholds and directions if they were provided
        if filtered_thresholds:
            filtered_mapping['thresholds'] = filtered_thresholds
        if filtered_threshold_directions:
            filtered_mapping['threshold_directions'] = filtered_threshold_directions

        self.logger.info(f"Filtered from {len(variables)} to {len(filtered_variables)} available pairs")
        return filtered_mapping

    def create_binary_labels(self, values: np.ndarray, threshold: float, direction: str) -> np.ndarray:
        """Threshold continuous values into binary labels."""
        if direction == 'less_than':
            return (values <= threshold).astype(int)
        elif direction == 'greater_than':
            return (values >= threshold).astype(int)
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'less_than' or 'greater_than'")

    def _write_fold_predictions(self, out_path, patient_ids, y_true, y_proba,
                                optimal_threshold=None) -> None:
        """Write one fold's predictions."""
        data = {
            'patient_id': patient_ids if patient_ids is not None else np.arange(len(y_true)),
            'true_label': y_true,
            'predicted_probability': y_proba,
        }
        if optimal_threshold is not None:
            data['predicted_label_at_optimal'] = (y_proba >= optimal_threshold).astype(int)
        pd.DataFrame(data).to_csv(out_path, index=False, float_format='%.6f')

    def cross_validation_analysis(self, X: np.ndarray, y: np.ndarray,
                                 patient_ids: np.ndarray = None,
                                 original_values: np.ndarray = None,
                                 compared_values: np.ndarray = None,
                                 X_test: np.ndarray = None,
                                 y_test: np.ndarray = None,
                                 test_patient_ids: np.ndarray = None,
                                 var_display_name: str = None) -> Dict:
        """Evaluate train, validation, and test folds."""
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        has_test = (X_test is not None) and (y_test is not None)
        safe_name = "fold" if var_display_name is None else "".join(
            c for c in var_display_name if c.isalnum() or c in (' ', '-', '_')
        ).rstrip().replace(' ', '_')

        train_metrics = {'auroc': [], 'auprc': [], 'f1': [], 'dor': [], 'tpr': [], 'tnr': [], 'fpr': [], 'fnr': []}
        val_metrics = {'auroc': [], 'auprc': [], 'f1': [], 'dor': [], 'tpr': [], 'tnr': [], 'fpr': [], 'fnr': []}
        test_metrics = {'auroc': [], 'auprc': [], 'f1': [], 'dor': [], 'tpr': [], 'tnr': [], 'fpr': [], 'fnr': []}
        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            self.logger.info(f"  Processing fold {fold_idx + 1}/{self.cv_folds}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Reshape for sklearn
            X_train_2d = X_train.reshape(-1, 1)
            X_val_2d = X_val.reshape(-1, 1)

            # Normalize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_2d)
            X_val_scaled = scaler.transform(X_val_2d)

            # Train model
            model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'
            )
            model.fit(X_train_scaled, y_train)

            # Predictions
            y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
            y_val_proba = model.predict_proba(X_val_scaled)[:, 1]

            # Calculate metrics for training set
            train_metrics['auroc'].append(roc_auc_score(y_train, y_train_proba))
            train_metrics['auprc'].append(average_precision_score(y_train, y_train_proba))

            # Use the training-set Youden threshold.
            fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_proba)
            youden_idx_train = np.argmax(tpr_train - fpr_train)
            optimal_threshold_train = thresholds_train[youden_idx_train]
            y_train_pred = (y_train_proba >= optimal_threshold_train).astype(int)
            train_metrics['f1'].append(f1_score(y_train, y_train_pred))

            # Calculate TPR, TNR, FPR, FNR at optimal threshold
            tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_train_pred).ravel()
            train_metrics['tpr'].append(tp_train / (tp_train + fn_train) if (tp_train + fn_train) > 0 else 0.0)
            train_metrics['tnr'].append(tn_train / (tn_train + fp_train) if (tn_train + fp_train) > 0 else 0.0)
            train_metrics['fpr'].append(fp_train / (fp_train + tn_train) if (fp_train + tn_train) > 0 else 0.0)
            train_metrics['fnr'].append(fn_train / (fn_train + tp_train) if (fn_train + tp_train) > 0 else 0.0)

            # DOR at optimal threshold
            if tp_train == 0 or tn_train == 0 or fp_train == 0 or fn_train == 0:
                tp_train_adj = tp_train + 0.5
                tn_train_adj = tn_train + 0.5
                fp_train_adj = fp_train + 0.5
                fn_train_adj = fn_train + 0.5
            else:
                tp_train_adj, tn_train_adj, fp_train_adj, fn_train_adj = tp_train, tn_train, fp_train, fn_train
            train_metrics['dor'].append((tp_train_adj * tn_train_adj) / (fp_train_adj * fn_train_adj))

            # Calculate metrics for validation set
            val_auroc = roc_auc_score(y_val, y_val_proba)
            val_metrics['auroc'].append(val_auroc)
            val_metrics['auprc'].append(average_precision_score(y_val, y_val_proba))

            # Use the validation-set Youden threshold.
            fpr_val, tpr_val, thresholds_val = roc_curve(y_val, y_val_proba)
            youden_idx_val = np.argmax(tpr_val - fpr_val)
            optimal_threshold_val = thresholds_val[youden_idx_val]
            y_val_pred = (y_val_proba >= optimal_threshold_val).astype(int)
            val_metrics['f1'].append(f1_score(y_val, y_val_pred))

            # Calculate TPR, TNR, FPR, FNR at optimal threshold
            tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_val, y_val_pred).ravel()
            val_metrics['tpr'].append(tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0.0)
            val_metrics['tnr'].append(tn_val / (tn_val + fp_val) if (tn_val + fp_val) > 0 else 0.0)
            val_metrics['fpr'].append(fp_val / (fp_val + tn_val) if (fp_val + tn_val) > 0 else 0.0)
            val_metrics['fnr'].append(fn_val / (fn_val + tp_val) if (fn_val + tp_val) > 0 else 0.0)

            # DOR at optimal threshold
            if tp_val == 0 or tn_val == 0 or fp_val == 0 or fn_val == 0:
                tp_val_adj = tp_val + 0.5
                tn_val_adj = tn_val + 0.5
                fp_val_adj = fp_val + 0.5
                fn_val_adj = fn_val + 0.5
            else:
                tp_val_adj, tn_val_adj, fp_val_adj, fn_val_adj = tp_val, tn_val, fp_val, fn_val
            val_metrics['dor'].append((tp_val_adj * tn_val_adj) / (fp_val_adj * fn_val_adj))

            # ----- External test predictions & metrics (fold-level) -----
            test_auroc = np.nan
            test_optimal_threshold = np.nan
            y_test_proba = None
            if has_test:
                X_test_2d = X_test.reshape(-1, 1)
                X_test_scaled = scaler.transform(X_test_2d)
                y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
                test_auroc = roc_auc_score(y_test, y_test_proba)
                test_metrics['auroc'].append(test_auroc)
                test_metrics['auprc'].append(average_precision_score(y_test, y_test_proba))

                fpr_te, tpr_te, thresholds_te = roc_curve(y_test, y_test_proba)
                youden_idx_te = np.argmax(tpr_te - fpr_te)
                test_optimal_threshold = thresholds_te[youden_idx_te]
                y_test_pred = (y_test_proba >= test_optimal_threshold).astype(int)
                test_metrics['f1'].append(f1_score(y_test, y_test_pred))
                tn_te, fp_te, fn_te, tp_te = confusion_matrix(y_test, y_test_pred).ravel()
                test_metrics['tpr'].append(tp_te / (tp_te + fn_te) if (tp_te + fn_te) > 0 else 0.0)
                test_metrics['tnr'].append(tn_te / (tn_te + fp_te) if (tn_te + fp_te) > 0 else 0.0)
                test_metrics['fpr'].append(fp_te / (fp_te + tn_te) if (fp_te + tn_te) > 0 else 0.0)
                test_metrics['fnr'].append(fn_te / (fn_te + tp_te) if (fn_te + tp_te) > 0 else 0.0)
                if tp_te == 0 or tn_te == 0 or fp_te == 0 or fn_te == 0:
                    tp_te_adj, tn_te_adj, fp_te_adj, fn_te_adj = tp_te + 0.5, tn_te + 0.5, fp_te + 0.5, fn_te + 0.5
                else:
                    tp_te_adj, tn_te_adj, fp_te_adj, fn_te_adj = tp_te, tn_te, fp_te, fn_te
                test_metrics['dor'].append((tp_te_adj * tn_te_adj) / (fp_te_adj * fn_te_adj))

                # Write per-fold external test predictions
                out_test = self.output_dir / f"fold_{fold_idx}_test_{safe_name}.csv"
                self._write_fold_predictions(
                    out_test, test_patient_ids, y_test, y_test_proba, test_optimal_threshold
                )

            # ----- Per-fold train & val prediction CSVs -----
            if patient_ids is not None:
                out_train = self.output_dir / f"fold_{fold_idx}_train_{safe_name}.csv"
                self._write_fold_predictions(
                    out_train, patient_ids[train_idx], y_train, y_train_proba, optimal_threshold_train
                )
                out_val = self.output_dir / f"fold_{fold_idx}_val_{safe_name}.csv"
                self._write_fold_predictions(
                    out_val, patient_ids[val_idx], y_val, y_val_proba, optimal_threshold_val
                )

            # Store fold-level results for median fold selection
            fold_result = {
                'fold_idx': fold_idx,
                'val_auroc': val_auroc,
                'test_auroc': test_auroc,
                'val_indices': val_idx,
                'y_val_true': y_val,
                'y_val_proba': y_val_proba,
                'optimal_threshold': optimal_threshold_val,
                'test_optimal_threshold': test_optimal_threshold,
            }

            # Add patient IDs and values if provided
            if patient_ids is not None:
                fold_result['val_patient_ids'] = patient_ids[val_idx]
            if original_values is not None:
                fold_result['val_original_values'] = original_values[val_idx]
            if compared_values is not None:
                fold_result['val_compared_values'] = compared_values[val_idx]

            fold_results.append(fold_result)

        # Calculate mean and std
        results = {
            'train': {
                'auroc_mean': np.mean(train_metrics['auroc']),
                'auroc_std': np.std(train_metrics['auroc'], ddof=1),
                'auprc_mean': np.mean(train_metrics['auprc']),
                'auprc_std': np.std(train_metrics['auprc'], ddof=1),
                'f1_mean': np.mean(train_metrics['f1']),
                'f1_std': np.std(train_metrics['f1'], ddof=1),
                'dor_mean': np.mean(train_metrics['dor']),
                'dor_std': np.std(train_metrics['dor'], ddof=1),
                'tpr_mean': np.mean(train_metrics['tpr']),
                'tpr_std': np.std(train_metrics['tpr'], ddof=1),
                'tnr_mean': np.mean(train_metrics['tnr']),
                'tnr_std': np.std(train_metrics['tnr'], ddof=1),
                'fpr_mean': np.mean(train_metrics['fpr']),
                'fpr_std': np.std(train_metrics['fpr'], ddof=1),
                'fnr_mean': np.mean(train_metrics['fnr']),
                'fnr_std': np.std(train_metrics['fnr'], ddof=1)
            },
            'val': {
                'auroc_mean': np.mean(val_metrics['auroc']),
                'auroc_std': np.std(val_metrics['auroc'], ddof=1),
                'auprc_mean': np.mean(val_metrics['auprc']),
                'auprc_std': np.std(val_metrics['auprc'], ddof=1),
                'f1_mean': np.mean(val_metrics['f1']),
                'f1_std': np.std(val_metrics['f1'], ddof=1),
                'dor_mean': np.mean(val_metrics['dor']),
                'dor_std': np.std(val_metrics['dor'], ddof=1),
                'tpr_mean': np.mean(val_metrics['tpr']),
                'tpr_std': np.std(val_metrics['tpr'], ddof=1),
                'tnr_mean': np.mean(val_metrics['tnr']),
                'tnr_std': np.std(val_metrics['tnr'], ddof=1),
                'fpr_mean': np.mean(val_metrics['fpr']),
                'fpr_std': np.std(val_metrics['fpr'], ddof=1),
                'fnr_mean': np.mean(val_metrics['fnr']),
                'fnr_std': np.std(val_metrics['fnr'], ddof=1)
            },
            'fold_results': fold_results,
        }

        if has_test and len(test_metrics['auroc']) > 0:
            results['test'] = {
                'auroc_mean': np.mean(test_metrics['auroc']),
                'auroc_std': np.std(test_metrics['auroc'], ddof=1),
                'auprc_mean': np.mean(test_metrics['auprc']),
                'auprc_std': np.std(test_metrics['auprc'], ddof=1),
                'f1_mean': np.mean(test_metrics['f1']),
                'f1_std': np.std(test_metrics['f1'], ddof=1),
                'dor_mean': np.mean(test_metrics['dor']),
                'dor_std': np.std(test_metrics['dor'], ddof=1),
                'tpr_mean': np.mean(test_metrics['tpr']),
                'tpr_std': np.std(test_metrics['tpr'], ddof=1),
                'tnr_mean': np.mean(test_metrics['tnr']),
                'tnr_std': np.std(test_metrics['tnr'], ddof=1),
                'fpr_mean': np.mean(test_metrics['fpr']),
                'fpr_std': np.std(test_metrics['fpr'], ddof=1),
                'fnr_mean': np.mean(test_metrics['fnr']),
                'fnr_std': np.std(test_metrics['fnr'], ddof=1),
                'raw': test_metrics,
            }
            results['train_raw'] = train_metrics
            results['val_raw'] = val_metrics

        return results

    def calculate_metrics_with_bootstrap_ci(self, X: np.ndarray, y: np.ndarray, alpha: float = 0.05,
                                           patient_ids: np.ndarray = None,
                                           original_values: np.ndarray = None,
                                           compared_values: np.ndarray = None) -> Dict:
        """Calculate metrics with bootstrap confidence intervals on full dataset."""
        # Reshape and normalize
        X_2d = X.reshape(-1, 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_2d)

        # Train model on full dataset
        model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            class_weight='balanced'
        )
        model.fit(X_scaled, y)
        y_proba = model.predict_proba(X_scaled)[:, 1]

        # Calculate point estimates
        auroc = roc_auc_score(y, y_proba)
        auprc = average_precision_score(y, y_proba)

        # Use the full-data Youden threshold.
        fpr, tpr, thresholds = roc_curve(y, y_proba)
        youden_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[youden_idx]
        y_pred = (y_proba >= optimal_threshold).astype(int)
        f1 = f1_score(y, y_pred)

        # Calculate TPR, TNR, FPR, FNR at optimal threshold
        tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y, y_pred).ravel()
        tpr_opt = tp_opt / (tp_opt + fn_opt) if (tp_opt + fn_opt) > 0 else 0.0
        tnr_opt = tn_opt / (tn_opt + fp_opt) if (tn_opt + fp_opt) > 0 else 0.0
        fpr_opt = fp_opt / (fp_opt + tn_opt) if (fp_opt + tn_opt) > 0 else 0.0
        fnr_opt = fn_opt / (fn_opt + tp_opt) if (fn_opt + tp_opt) > 0 else 0.0

        # DOR at optimal threshold
        if tp_opt == 0 or tn_opt == 0 or fp_opt == 0 or fn_opt == 0:
            tp_opt_adj = tp_opt + 0.5
            tn_opt_adj = tn_opt + 0.5
            fp_opt_adj = fp_opt + 0.5
            fn_opt_adj = fn_opt + 0.5
        else:
            tp_opt_adj, tn_opt_adj, fp_opt_adj, fn_opt_adj = tp_opt, tn_opt, fp_opt, fn_opt
        dor = (tp_opt_adj * tn_opt_adj) / (fp_opt_adj * fn_opt_adj)

        # Bootstrap for CIs
        self.logger.info(f"  Computing bootstrap CIs ({self.n_bootstrap} iterations)...")
        bootstrap_auroc = []
        bootstrap_auprc = []
        bootstrap_f1 = []
        bootstrap_dor = []
        bootstrap_tpr = []
        bootstrap_tnr = []
        bootstrap_fpr = []
        bootstrap_fnr = []

        rng = np.random.RandomState(self.random_state)

        for i in range(self.n_bootstrap):
            # Bootstrap sample
            indices = rng.choice(len(y), size=len(y), replace=True)

            # Skip if only one class present
            if len(np.unique(y[indices])) < 2:
                continue

            X_boot = X_scaled[indices]
            y_boot = y[indices]

            # Train model on bootstrap sample
            model_boot = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'
            )
            model_boot.fit(X_boot, y_boot)
            y_boot_proba = model_boot.predict_proba(X_boot)[:, 1]

            # Calculate metrics
            bootstrap_auroc.append(roc_auc_score(y_boot, y_boot_proba))
            bootstrap_auprc.append(average_precision_score(y_boot, y_boot_proba))

            # Reuse the full-data Youden threshold.
            y_boot_pred = (y_boot_proba >= optimal_threshold).astype(int)
            bootstrap_f1.append(f1_score(y_boot, y_boot_pred))

            # Calculate TPR, TNR, FPR, FNR
            tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_boot, y_boot_pred).ravel()
            bootstrap_tpr.append(tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0)
            bootstrap_tnr.append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0)
            bootstrap_fpr.append(fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0.0)
            bootstrap_fnr.append(fn_b / (fn_b + tp_b) if (fn_b + tp_b) > 0 else 0.0)

            # DOR at optimal threshold
            if tp_b == 0 or tn_b == 0 or fp_b == 0 or fn_b == 0:
                tp_b_adj = tp_b + 0.5
                tn_b_adj = tn_b + 0.5
                fp_b_adj = fp_b + 0.5
                fn_b_adj = fn_b + 0.5
            else:
                tp_b_adj, tn_b_adj, fp_b_adj, fn_b_adj = tp_b, tn_b, fp_b, fn_b
            bootstrap_dor.append((tp_b_adj * tn_b_adj) / (fp_b_adj * fn_b_adj))

        # Calculate percentile CIs
        results = {
            'auroc': auroc,
            'auroc_ci_lower': np.percentile(bootstrap_auroc, alpha/2 * 100),
            'auroc_ci_upper': np.percentile(bootstrap_auroc, (1 - alpha/2) * 100),
            'auprc': auprc,
            'auprc_ci_lower': np.percentile(bootstrap_auprc, alpha/2 * 100),
            'auprc_ci_upper': np.percentile(bootstrap_auprc, (1 - alpha/2) * 100),
            'f1': f1,
            'f1_ci_lower': np.percentile(bootstrap_f1, alpha/2 * 100),
            'f1_ci_upper': np.percentile(bootstrap_f1, (1 - alpha/2) * 100),
            'f1_optimal_threshold': optimal_threshold,
            'dor': dor,
            'dor_ci_lower': np.percentile(bootstrap_dor, alpha/2 * 100),
            'dor_ci_upper': np.percentile(bootstrap_dor, (1 - alpha/2) * 100),
            'tpr': tpr_opt,
            'tpr_ci_lower': np.percentile(bootstrap_tpr, alpha/2 * 100),
            'tpr_ci_upper': np.percentile(bootstrap_tpr, (1 - alpha/2) * 100),
            'tnr': tnr_opt,
            'tnr_ci_lower': np.percentile(bootstrap_tnr, alpha/2 * 100),
            'tnr_ci_upper': np.percentile(bootstrap_tnr, (1 - alpha/2) * 100),
            'fpr': fpr_opt,
            'fpr_ci_lower': np.percentile(bootstrap_fpr, alpha/2 * 100),
            'fpr_ci_upper': np.percentile(bootstrap_fpr, (1 - alpha/2) * 100),
            'fnr': fnr_opt,
            'fnr_ci_lower': np.percentile(bootstrap_fnr, alpha/2 * 100),
            'fnr_ci_upper': np.percentile(bootstrap_fnr, (1 - alpha/2) * 100),
            'y_proba': y_proba,
            'patient_ids': patient_ids,
            'original_values': original_values,
            'compared_values': compared_values
        }

        return results

    def export_full_predictions(self, full_results: Dict, y_true: np.ndarray,
                               var_name: str, optimal_threshold: float) -> None:
        """Export predictions from full dataset model to CSV."""
        # Check if patient IDs are available
        if full_results.get('patient_ids') is None:
            self.logger.warning("Cannot export predictions: patient IDs not available")
            return

        # Create output dataframe
        output_data = {
            'patient_id': full_results['patient_ids'],
            'true_label': y_true,
            'predicted_probability': full_results['y_proba'],
            'predicted_label_at_optimal': (full_results['y_proba'] >= optimal_threshold).astype(int),
            'predicted_label_at_05': (full_results['y_proba'] >= 0.5).astype(int)
        }

        # Add original and compared values if available
        if full_results.get('original_values') is not None:
            output_data['original_value'] = full_results['original_values']
        if full_results.get('compared_values') is not None:
            output_data['compared_value'] = full_results['compared_values']

        df = pd.DataFrame(output_data)

        # Save to CSV
        safe_name = "".join(c for c in var_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
        output_path = self.output_dir / f"predictions_{safe_name}.csv"
        df.to_csv(output_path, index=False, float_format='%.6f')

        self.logger.info(f"  Full dataset predictions saved: predictions_{safe_name}.csv")
        self.logger.info(f"  Total samples: {len(df)}")

    def export_confusion_matrix(self, full_results: Dict, y_true: np.ndarray,
                               var_name: str, optimal_threshold: float) -> None:
        """Export confusion matrix and derived metrics to CSV."""
        y_proba = full_results['y_proba']

        # Predictions at optimal threshold
        y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
        tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_true, y_pred_optimal).ravel()

        # Predictions at threshold 0.5
        y_pred_05 = (y_proba >= 0.5).astype(int)
        tn_05, fp_05, fn_05, tp_05 = confusion_matrix(y_true, y_pred_05).ravel()

        # Calculate derived metrics for optimal threshold
        sensitivity_opt = tp_opt / (tp_opt + fn_opt) if (tp_opt + fn_opt) > 0 else 0.0
        specificity_opt = tn_opt / (tn_opt + fp_opt) if (tn_opt + fp_opt) > 0 else 0.0
        precision_opt = tp_opt / (tp_opt + fp_opt) if (tp_opt + fp_opt) > 0 else 0.0
        npv_opt = tn_opt / (tn_opt + fn_opt) if (tn_opt + fn_opt) > 0 else 0.0
        accuracy_opt = (tp_opt + tn_opt) / (tp_opt + tn_opt + fp_opt + fn_opt)

        # Calculate derived metrics for threshold 0.5
        sensitivity_05 = tp_05 / (tp_05 + fn_05) if (tp_05 + fn_05) > 0 else 0.0
        specificity_05 = tn_05 / (tn_05 + fp_05) if (tn_05 + fp_05) > 0 else 0.0
        precision_05 = tp_05 / (tp_05 + fp_05) if (tp_05 + fp_05) > 0 else 0.0
        npv_05 = tn_05 / (tn_05 + fn_05) if (tn_05 + fn_05) > 0 else 0.0
        accuracy_05 = (tp_05 + tn_05) / (tp_05 + tn_05 + fp_05 + fn_05)

        # Create confusion matrix report
        report_data = {
            'Threshold': ['Optimal', '0.5'],
            'Threshold_Value': [f"{optimal_threshold:.4f}", "0.5000"],
            'True_Positives': [tp_opt, tp_05],
            'False_Positives': [fp_opt, fp_05],
            'True_Negatives': [tn_opt, tn_05],
            'False_Negatives': [fn_opt, fn_05],
            'Sensitivity_TPR': [f"{sensitivity_opt:.4f}", f"{sensitivity_05:.4f}"],
            'Specificity_TNR': [f"{specificity_opt:.4f}", f"{specificity_05:.4f}"],
            'Precision_PPV': [f"{precision_opt:.4f}", f"{precision_05:.4f}"],
            'NPV': [f"{npv_opt:.4f}", f"{npv_05:.4f}"],
            'Accuracy': [f"{accuracy_opt:.4f}", f"{accuracy_05:.4f}"]
        }

        df = pd.DataFrame(report_data)

        # Save to CSV
        safe_name = "".join(c for c in var_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
        output_path = self.output_dir / f"confusion_matrix_{safe_name}.csv"
        df.to_csv(output_path, index=False)

        self.logger.info(f"  Confusion matrix saved: confusion_matrix_{safe_name}.csv")

    def export_cv_results_detailed_and_summary(self, cv_results: Dict, var_display_name: str) -> None:
        """Export fold metrics and cross-fold summaries."""
        if 'test' not in cv_results:
            self.logger.info("  Skipping cv_results_detailed export (no external test data provided)")
            return

        safe_name = "".join(
            c for c in var_display_name if c.isalnum() or c in (' ', '-', '_')
        ).rstrip().replace(' ', '_')

        train_raw = cv_results['train_raw']
        val_raw = cv_results['val_raw']
        test_raw = cv_results['test']['raw']
        n_folds = len(train_raw['auroc'])

        rows = []
        metric_keys = ['auroc', 'auprc', 'f1', 'tpr', 'tnr', 'fpr', 'fnr', 'dor']
        for i in range(n_folds):
            row = {'fold': i}
            for m in metric_keys:
                row[f'train_{m}'] = train_raw[m][i]
                row[f'internal_val_{m}'] = val_raw[m][i]
                row[f'external_val_{m}'] = test_raw[m][i]

            row['train_auc'] = row['train_auroc']
            row['internal_val_auc'] = row['internal_val_auroc']
            row['external_val_auc'] = row['external_val_auroc']
            # Fold prediction CSV paths.
            row['fold_train_csv'] = str(
                (self.output_dir / f"fold_{i}_train_{safe_name}.csv").resolve()
            )
            row['fold_val_csv'] = str(
                (self.output_dir / f"fold_{i}_val_{safe_name}.csv").resolve()
            )
            row['fold_test_csv'] = str(
                (self.output_dir / f"fold_{i}_test_{safe_name}.csv").resolve()
            )
            rows.append(row)

        detailed_df = pd.DataFrame(rows)
        detailed_path = self.output_dir / f"cv_results_detailed_{safe_name}.csv"
        detailed_df.to_csv(detailed_path, index=False, float_format='%.6f')
        self.logger.info(f"  CV detailed results saved: {detailed_path.name}")

        # Cross-fold summary
        summary_row = {'n_folds': n_folds, 'variable': var_display_name}
        for m in metric_keys:
            summary_row[f'train_{m}_mean'] = float(np.mean(train_raw[m]))
            summary_row[f'train_{m}_std'] = float(np.std(train_raw[m], ddof=1))
            summary_row[f'internal_val_{m}_mean'] = float(np.mean(val_raw[m]))
            summary_row[f'internal_val_{m}_std'] = float(np.std(val_raw[m], ddof=1))
            summary_row[f'external_val_{m}_mean'] = float(np.mean(test_raw[m]))
            summary_row[f'external_val_{m}_std'] = float(np.std(test_raw[m], ddof=1))
        summary_df = pd.DataFrame([summary_row])
        summary_path = self.output_dir / f"cv_results_summary_{safe_name}.csv"
        summary_df.to_csv(summary_path, index=False, float_format='%.6f')
        self.logger.info(f"  CV summary saved: {summary_path.name}")

    def plot_roc_curve(self, cv_results: Dict, full_results: Dict,
                      var_name: str, y_true: np.ndarray) -> None:
        """Plot ROC curves for training and validation sets."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get probabilities for full dataset
        y_proba_full = full_results['y_proba']

        # Calculate ROC curve for full dataset
        fpr_full, tpr_full, _ = roc_curve(y_true, y_proba_full)

        # Plot full dataset ROC
        ax.plot(fpr_full, tpr_full, 'b-', linewidth=2.5, alpha=0.7,
                label=f'Full Dataset (AUC = {full_results["auroc"]:.3f} '
                      f'[{full_results["auroc_ci_lower"]:.3f}, {full_results["auroc_ci_upper"]:.3f}])')

        # Plot random classifier
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.7, label='Random Classifier')

        # Styling
        ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        ax.set_title(f'ROC Curve: {var_name}\n'
                    f'Train AUC: {cv_results["train"]["auroc_mean"]:.3f}±{cv_results["train"]["auroc_std"]:.3f}, '
                    f'Val AUC: {cv_results["val"]["auroc_mean"]:.3f}±{cv_results["val"]["auroc_std"]:.3f}',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        plt.tight_layout()
        safe_name = "".join(c for c in var_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
        plt.savefig(self.output_dir / f"roc_curve_{safe_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"  ROC curve saved: roc_curve_{safe_name}.png")

    def plot_pr_curve(self, cv_results: Dict, full_results: Dict,
                     var_name: str, y_true: np.ndarray) -> None:
        """Plot Precision-Recall curves for training and validation sets."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get probabilities for full dataset
        y_proba_full = full_results['y_proba']

        # Calculate PR curve for full dataset
        precision_full, recall_full, _ = precision_recall_curve(y_true, y_proba_full)

        # Baseline (prevalence)
        baseline = np.sum(y_true) / len(y_true)

        # Plot full dataset PR curve
        ax.plot(recall_full, precision_full, 'b-', linewidth=2.5, alpha=0.7,
                label=f'Full Dataset (AUPRC = {full_results["auprc"]:.3f} '
                      f'[{full_results["auprc_ci_lower"]:.3f}, {full_results["auprc_ci_upper"]:.3f}])')

        # Plot baseline
        ax.axhline(y=baseline, color='r', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'No Skill (baseline = {baseline:.3f})')

        # Styling
        ax.set_xlabel('Recall (Sensitivity)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Precision (PPV)', fontsize=14, fontweight='bold')
        ax.set_title(f'Precision-Recall Curve: {var_name}\n'
                    f'Train AUPRC: {cv_results["train"]["auprc_mean"]:.3f}±{cv_results["train"]["auprc_std"]:.3f}, '
                    f'Val AUPRC: {cv_results["val"]["auprc_mean"]:.3f}±{cv_results["val"]["auprc_std"]:.3f}',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        plt.tight_layout()
        safe_name = "".join(c for c in var_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
        plt.savefig(self.output_dir / f"pr_curve_{safe_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"  PR curve saved: pr_curve_{safe_name}.png")

    def analyze_column_pairs(self, column_mapping: Dict[str, List[str]]) -> None:
        """Analyze multiple column pairs and generate results."""
        self.logger.info("-" * 40)
        self.logger.info("COLUMN PAIR ANALYSIS")
        self.logger.info("-" * 40)

        # Merge data
        merged_df = self.analyze_id_overlap()

        # Optional: merge held-out test set
        test_merged_df = None
        test_id_col = None
        if self.has_test_set:
            test_merged_df = self.analyze_id_overlap_test()
            test_id_col = self.original_id_col if self.original_id_col in test_merged_df.columns else f"{self.original_id_col}_original"

        # Determine ID column name after merge
        id_col = self.original_id_col if self.original_id_col in merged_df.columns else f"{self.original_id_col}_original"

        # Filter to only available columns
        filtered_mapping = self.filter_available_columns(column_mapping, merged_df)

        if len(filtered_mapping['Variables']) == 0:
            self.logger.error("No matching column pairs found!")
            return

        variables = filtered_mapping['Variables']
        original_cols = filtered_mapping['original_cols']
        compared_cols = filtered_mapping['compared_cols']
        thresholds = filtered_mapping.get('thresholds', [])
        threshold_directions = filtered_mapping.get('threshold_directions', [])

        self.logger.info(f"Total variable pairs to analyze: {len(variables)}")

        results_list = []

        # Require a threshold and direction per variable.
        if not thresholds or not threshold_directions:
            self.logger.error("Missing thresholds or threshold_directions in column_mapping!")
            self.logger.error("Please provide 'thresholds' and 'threshold_directions' lists in column_mapping")
            return

        for i, (var_name, orig_col, comp_col, threshold, direction) in enumerate(
            zip(variables, original_cols, compared_cols, thresholds, threshold_directions)
        ):
            threshold_str = str(threshold).replace('.', 'p')
            var_display_name = f"{var_name}_{threshold_str}"

            self.logger.info(f"\n--- Processing {i+1}/{len(variables)}: {var_display_name} ---")
            self.logger.info(f"Variable: {var_name}, Threshold: {threshold} ({direction})")
            self.logger.info(f"Original column: '{orig_col}'")
            self.logger.info(f"Compared column: '{comp_col}'")

            try:
                # Get data
                orig_data = merged_df[orig_col]
                comp_data = merged_df[comp_col]
                id_data = merged_df[id_col]

                self.logger.info(f"Original data - Total: {len(orig_data)}, Missing: {orig_data.isna().sum()}")
                self.logger.info(f"Compared data - Total: {len(comp_data)}, Missing: {comp_data.isna().sum()}")

                # Remove NaN values and track patient IDs
                mask = ~(orig_data.isna() | comp_data.isna())
                orig_clean = orig_data.loc[mask].values
                comp_clean = comp_data.loc[mask].values
                patient_ids_clean = id_data.loc[mask].values

                self.logger.info(f"Valid pairs after removing NaN: {len(orig_clean)}")

                if len(orig_clean) < 50:
                    self.logger.warning(f"Insufficient data for {var_display_name}: {len(orig_clean)} samples")
                    continue

                # Threshold original values into binary labels.
                y = self.create_binary_labels(orig_clean, threshold, direction)
                X = comp_clean

                # Check class distribution
                n_diseased = np.sum(y == 1)
                n_healthy = np.sum(y == 0)

                # Update logging based on direction
                if direction == 'less_than':
                    self.logger.info(f"Class distribution - Diseased (<{threshold}): {n_diseased}, "
                                   f"Healthy (>={threshold}): {n_healthy}")
                else:  # greater_than
                    self.logger.info(f"Class distribution - Diseased (>={threshold}): {n_diseased}, "
                                   f"Healthy (<{threshold}): {n_healthy}")

                if n_diseased < 10 or n_healthy < 10:
                    self.logger.warning(f"Insufficient samples per class for {var_display_name}")
                    continue

                # Extract this variable's held-out test slice.
                X_test_arr = None
                y_test_arr = None
                test_pid_arr = None
                if test_merged_df is not None:
                    if orig_col in test_merged_df.columns and comp_col in test_merged_df.columns:
                        t_orig = test_merged_df[orig_col]
                        t_comp = test_merged_df[comp_col]
                        t_id = test_merged_df[test_id_col] if test_id_col is not None else None
                        if t_id is not None:
                            t_mask = ~(t_orig.isna() | t_comp.isna() | t_id.isna())
                            test_pid_arr = t_id.loc[t_mask].values
                        else:
                            t_mask = ~(t_orig.isna() | t_comp.isna())
                        t_orig_clean = t_orig.loc[t_mask].values
                        t_comp_clean = t_comp.loc[t_mask].values
                        y_test_arr = self.create_binary_labels(t_orig_clean, threshold, direction)
                        X_test_arr = t_comp_clean
                        self.logger.info(f"External test samples: {len(y_test_arr)} "
                                         f"(pos={int(np.sum(y_test_arr == 1))}, neg={int(np.sum(y_test_arr == 0))})")
                    else:
                        self.logger.warning(f"Test merge missing columns for {var_display_name}; skipping test scoring")

                # Run cross-validation and export patient-level predictions.
                self.logger.info(f"Performing {self.cv_folds}-fold cross-validation...")
                cv_results = self.cross_validation_analysis(
                    X, y,
                    patient_ids=patient_ids_clean,
                    original_values=orig_clean,
                    compared_values=comp_clean,
                    X_test=X_test_arr,
                    y_test=y_test_arr,
                    test_patient_ids=test_pid_arr,
                    var_display_name=var_display_name,
                )

                # Export per-fold detailed + summary when test set present
                if 'test' in cv_results:
                    self.export_cv_results_detailed_and_summary(cv_results, var_display_name)

                if self.has_test_set:
                    self.logger.info("Skipping bootstrap CI on training merge (external test provided)")
                    full_results = None
                else:
                    # Calculate metrics with bootstrap CI on full dataset
                    self.logger.info(f"Calculating full dataset metrics with bootstrap CIs...")
                    full_results = self.calculate_metrics_with_bootstrap_ci(
                        X, y,
                        patient_ids=patient_ids_clean,
                        original_values=orig_clean,
                        compared_values=comp_clean
                    )

                # Log results
                self.logger.info("")
                self.logger.info(f"Results for {var_display_name}:")
                self.logger.info(f"  Training:   AUROC: {cv_results['train']['auroc_mean']:.3f}±{cv_results['train']['auroc_std']:.3f}  "
                               f"AUPRC: {cv_results['train']['auprc_mean']:.3f}±{cv_results['train']['auprc_std']:.3f}  "
                               f"F1: {cv_results['train']['f1_mean']:.3f}±{cv_results['train']['f1_std']:.3f}  "
                               f"DOR: {cv_results['train']['dor_mean']:.1f}±{cv_results['train']['dor_std']:.1f}")
                self.logger.info(f"  Validation: AUROC: {cv_results['val']['auroc_mean']:.3f}±{cv_results['val']['auroc_std']:.3f}  "
                               f"AUPRC: {cv_results['val']['auprc_mean']:.3f}±{cv_results['val']['auprc_std']:.3f}  "
                               f"F1: {cv_results['val']['f1_mean']:.3f}±{cv_results['val']['f1_std']:.3f}  "
                               f"DOR: {cv_results['val']['dor_mean']:.1f}±{cv_results['val']['dor_std']:.1f}")
                if 'test' in cv_results:
                    self.logger.info(f"  External:   AUROC: {cv_results['test']['auroc_mean']:.3f}±{cv_results['test']['auroc_std']:.3f}  "
                                   f"AUPRC: {cv_results['test']['auprc_mean']:.3f}±{cv_results['test']['auprc_std']:.3f}  "
                                   f"F1: {cv_results['test']['f1_mean']:.3f}±{cv_results['test']['f1_std']:.3f}  "
                                   f"DOR: {cv_results['test']['dor_mean']:.1f}±{cv_results['test']['dor_std']:.1f}")
                if full_results is not None:
                    self.logger.info(f"  Full (CI):  AUROC: {full_results['auroc']:.3f} [{full_results['auroc_ci_lower']:.3f}, {full_results['auroc_ci_upper']:.3f}]  "
                                   f"AUPRC: {full_results['auprc']:.3f} [{full_results['auprc_ci_lower']:.3f}, {full_results['auprc_ci_upper']:.3f}]  "
                                   f"F1: {full_results['f1']:.3f} [{full_results['f1_ci_lower']:.3f}, {full_results['f1_ci_upper']:.3f}]  "
                                   f"DOR: {full_results['dor']:.1f} [{full_results['dor_ci_lower']:.1f}, {full_results['dor_ci_upper']:.1f}]")

                # Export bootstrap artifacts only when available.
                if full_results is not None:
                    self.plot_roc_curve(cv_results, full_results, var_display_name, y)
                    self.plot_pr_curve(cv_results, full_results, var_display_name, y)
                    self.export_full_predictions(full_results, y, var_display_name, full_results['f1_optimal_threshold'])
                    self.export_confusion_matrix(full_results, y, var_display_name, full_results['f1_optimal_threshold'])

                # Store results
                results_list.append({
                    'variable': var_display_name,
                    'n_samples': len(orig_clean),
                    'n_diseased': n_diseased,
                    'n_healthy': n_healthy,
                    'threshold': threshold,
                    'threshold_direction': direction,
                    'cv_folds': self.cv_folds,
                    'cv_results': cv_results,
                    'full_results': full_results
                })

            except Exception as e:
                self.logger.error(f"Error processing {var_display_name}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

        # Export summary CSV
        if len(results_list) > 0:
            self.export_summary_csv(results_list)

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("PREDICTION ANALYSIS COMPLETED")
        self.logger.info("=" * 80)

    def export_summary_csv(self, results_list: List[Dict]) -> None:
        """Export summary metrics with confidence intervals."""
        self.logger.info("-" * 40)
        self.logger.info("EXPORTING SUMMARY CSV")
        self.logger.info("-" * 40)

        summary_data = []
        for res in results_list:
            cv = res['cv_results']
            full = res['full_results']

            row = {
                'Variable': res['variable'],
                'threshold': res['threshold'],
                'threshold_direction': res['threshold_direction'],
                'n_samples': res['n_samples'],
                'n_diseased': res['n_diseased'],
                'n_healthy': res['n_healthy'],
                'cv_folds': res['cv_folds'],

                # Training metrics use mean and standard deviation.
                'AUROC_train': f"{cv['train']['auroc_mean']:.4f}, {cv['train']['auroc_std']:.4f}",
                'AUPRC_train': f"{cv['train']['auprc_mean']:.4f}, {cv['train']['auprc_std']:.4f}",
                'F1_train': f"{cv['train']['f1_mean']:.4f}, {cv['train']['f1_std']:.4f}",
                'DOR_train': f"{cv['train']['dor_mean']:.4f}, {cv['train']['dor_std']:.4f}",
                'TPR_train': f"{cv['train']['tpr_mean']:.4f}, {cv['train']['tpr_std']:.4f}",
                'TNR_train': f"{cv['train']['tnr_mean']:.4f}, {cv['train']['tnr_std']:.4f}",
                'FPR_train': f"{cv['train']['fpr_mean']:.4f}, {cv['train']['fpr_std']:.4f}",
                'FNR_train': f"{cv['train']['fnr_mean']:.4f}, {cv['train']['fnr_std']:.4f}",

                # Validation metrics use mean and standard deviation.
                'AUROC_val': f"{cv['val']['auroc_mean']:.4f}, {cv['val']['auroc_std']:.4f}",
                'AUPRC_val': f"{cv['val']['auprc_mean']:.4f}, {cv['val']['auprc_std']:.4f}",
                'F1_val': f"{cv['val']['f1_mean']:.4f}, {cv['val']['f1_std']:.4f}",
                'DOR_val': f"{cv['val']['dor_mean']:.4f}, {cv['val']['dor_std']:.4f}",
                'TPR_val': f"{cv['val']['tpr_mean']:.4f}, {cv['val']['tpr_std']:.4f}",
                'TNR_val': f"{cv['val']['tnr_mean']:.4f}, {cv['val']['tnr_std']:.4f}",
                'FPR_val': f"{cv['val']['fpr_mean']:.4f}, {cv['val']['fpr_std']:.4f}",
                'FNR_val': f"{cv['val']['fnr_mean']:.4f}, {cv['val']['fnr_std']:.4f}",
            }

            if 'test' in cv:
                # External metrics use fold mean and standard deviation.
                for m in ['auroc', 'auprc', 'f1', 'dor', 'tpr', 'tnr', 'fpr', 'fnr']:
                    row[f'{m.upper()}_external_test'] = (
                        f"{cv['test'][f'{m}_mean']:.4f}, {cv['test'][f'{m}_std']:.4f}"
                    )

            if full is not None:
                # Full metrics use confidence intervals.
                row['AUROC_full'] = f"{full['auroc']:.4f} [{full['auroc_ci_lower']:.4f}, {full['auroc_ci_upper']:.4f}]"
                row['AUPRC_full'] = f"{full['auprc']:.4f} [{full['auprc_ci_lower']:.4f}, {full['auprc_ci_upper']:.4f}]"
                row['F1_full'] = f"{full['f1']:.4f} [{full['f1_ci_lower']:.4f}, {full['f1_ci_upper']:.4f}]"
                row['DOR_full'] = f"{full['dor']:.4f} [{full['dor_ci_lower']:.4f}, {full['dor_ci_upper']:.4f}]"
                row['TPR_full'] = f"{full['tpr']:.4f} [{full['tpr_ci_lower']:.4f}, {full['tpr_ci_upper']:.4f}]"
                row['TNR_full'] = f"{full['tnr']:.4f} [{full['tnr_ci_lower']:.4f}, {full['tnr_ci_upper']:.4f}]"
                row['FPR_full'] = f"{full['fpr']:.4f} [{full['fpr_ci_lower']:.4f}, {full['fpr_ci_upper']:.4f}]"
                row['FNR_full'] = f"{full['fnr']:.4f} [{full['fnr_ci_lower']:.4f}, {full['fnr_ci_upper']:.4f}]"
                row['F1_optimal_threshold'] = f"{full['f1_optimal_threshold']:.4f}"

            summary_data.append(row)

        df = pd.DataFrame(summary_data)
        output_path = self.output_dir / "prediction_analysis_summary.csv"

        # Check if file exists to determine append mode
        file_exists = output_path.exists()
        df.to_csv(output_path, mode='a', header=not file_exists, index=False)

        action = "appended to" if file_exists else "created"
        self.logger.info(f"Summary CSV {action}: {output_path}")
        self.logger.info(f"  - Rows: {len(df)}")
        self.logger.info(f"  - Columns: {len(df.columns)}")


def main():
    parser = argparse.ArgumentParser(
        description='Prediction Analysis for Binary Classification Performance Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--original_path', type=str, required=True,
                       help='Path to GROUND-TRUTH CSV (y_true source, shared between train and test evaluation)')
    parser.add_argument('--compared_path', type=str, required=True,
                       help='Path to TRAINING predictions CSV (X source for CV folds)')
    parser.add_argument('--test_compared_path', type=str, required=True,
                       help='Path to TEST predictions CSV')
    parser.add_argument('--test_original_path', type=str, default=None,
                       help='Optional: separate ground-truth CSV for the test merge (default: reuse --original_path)')

    parser.add_argument('--original_id_col', type=str, default='patient_id',
                       help='ID column name in original table (default: patient_id)')
    parser.add_argument('--compared_id_col', type=str, default='patient_id',
                       help='ID column name in compared table (default: patient_id)')

    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for plots and results')

    # Single-variable CLI overrides.
    parser.add_argument('--variable', type=str, default=None,
                       help='Variable display name (overrides built-in mapping when set with --original_col/--compared_col/--threshold/--direction)')
    parser.add_argument('--original_col', type=str, default=None,
                       help='Original column name (single-variable override)')
    parser.add_argument('--compared_col', type=str, default=None,
                       help='Compared column name (single-variable override)')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Disease threshold (single-variable override)')
    parser.add_argument('--direction', type=str, default=None, choices=[None, 'less_than', 'greater_than'],
                       help='Threshold direction (single-variable override)')

    parser.add_argument('--n_bootstrap', type=int, default=1000,
                       help='Number of bootstrap iterations for CIs (default: 1000; skipped when test set provided)')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    parser.add_argument('--instance_col', type=str, default=None,
                       help='Instance column name for deduplication (default: None)')

    args = parser.parse_args()

    # Default LVEF and wall-thickness tasks.
    all_column_mapping = {
        'Variables': ['LVEF', 'LVEF', 'Wall thickness max', 'Wall thickness max'],
        'original_cols': ['lv_ef_fr0_percent', 'lv_ef_fr0_percent', 'sax_wt_max_fr0', 'sax_wt_max_fr0'],
        'compared_cols': ['lv_ef_fr0_percent', 'lv_ef_fr0_percent', 'sax_wt_max_fr0', 'sax_wt_max_fr0'],
        'thresholds': [45, 50, 13, 15],
        'threshold_directions': ['less_than', 'less_than', 'greater_than', 'greater_than'],
    }

    # Single-variable CLI override (used by the batch runner)
    if (args.variable is not None and args.original_col is not None
            and args.compared_col is not None and args.threshold is not None
            and args.direction is not None):
        all_column_mapping = {
            'Variables': [args.variable],
            'original_cols': [args.original_col],
            'compared_cols': [args.compared_col],
            'thresholds': [args.threshold],
            'threshold_directions': [args.direction],
        }

    # Resolve optional test paths (empty string disables)
    test_original = args.test_original_path if args.test_original_path else None
    test_compared = args.test_compared_path if args.test_compared_path else None

    # Initialize analyzer
    analyzer = VisionECGPhenotypeAnalyzer(
        original_path=args.original_path,
        compared_path=args.compared_path,
        original_id_col=args.original_id_col,
        compared_id_col=args.compared_id_col,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        instance_col=args.instance_col,
        test_original_path=test_original,
        test_compared_path=test_compared,
    )

    # Perform analysis
    analyzer.analyze_column_pairs(all_column_mapping)


if __name__ == "__main__":
    main()
