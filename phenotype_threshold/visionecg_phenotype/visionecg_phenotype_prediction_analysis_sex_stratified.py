"""Sex-stratified phenotype classification with logistic cross-validation."""

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


class VisionECGPhenotypeSexStratifiedAnalyzer:
    def __init__(self, original_path: str, compared_path: str,
                 original_id_col: str, compared_id_col: str,
                 sex_col: str = 'Sex',
                 male_value: int = 1,
                 female_value: int = 0,
                 output_dir: str = "prediction_analysis_sex_stratified_results",
                 n_bootstrap: int = 1000,
                 cv_folds: int = 5,
                 random_state: int = 42,
                 instance_col: str = None,
                 test_original_path: str = None,
                 test_compared_path: str = None):
        """Initialize the Sex-Stratified Prediction Analyzer."""
        self.original_path = original_path
        self.compared_path = compared_path
        self.original_id_col = original_id_col
        self.compared_id_col = compared_id_col
        self.sex_col = sex_col
        self.male_value = male_value
        self.female_value = female_value
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
        self.logger.info("SEX-STRATIFIED PREDICTION ANALYSIS STARTED")
        self.logger.info("=" * 80)
        self.load_data()

        if self.has_test_set:
            self.load_test_data()

    def setup_logging(self):
        """Setup comprehensive logging system."""
        # Create logger
        self.logger = logging.getLogger('VisionECGPhenotypeSexStratifiedAnalyzer')
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
        log_file = self.output_dir / f"prediction_analysis_sex_stratified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
        self.test_compared_df = pd.read_csv(self.test_compared_path)
        self.logger.info(f"Test compared loaded: {self.test_compared_df.shape[0]} rows, {self.test_compared_df.shape[1]} columns")
        if self.compared_id_col not in self.test_compared_df.columns:
            raise ValueError(f"ID column '{self.compared_id_col}' not found in test compared")

        if self.test_original_path is None or self.test_original_path == self.original_path:
            self.logger.info("Reusing training ground-truth (--original_path) for the test merge")
            self.test_original_df = self.original_df
        else:
            self.logger.info(f"Loading TEST original data from: {self.test_original_path}")
            self.test_original_df = pd.read_csv(self.test_original_path)
            self.logger.info(f"Test original loaded: {self.test_original_df.shape[0]} rows, {self.test_original_df.shape[1]} columns")
            if self.original_id_col not in self.test_original_df.columns:
                raise ValueError(f"ID column '{self.original_id_col}' not found in test original")

    def analyze_id_overlap_test(self) -> pd.DataFrame:
        """Merge held-out truth and predictions by patient ID."""
        self.logger.info("-" * 40)
        self.logger.info("TEST ID OVERLAP ANALYSIS")
        self.logger.info("-" * 40)

        original_ids = set(self.test_original_df[self.original_id_col].dropna())
        compared_ids = set(self.test_compared_df[self.compared_id_col].dropna())
        overlapping_ids = original_ids.intersection(compared_ids)
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
        male_thresholds = column_mapping.get('male_thresholds', [])
        female_thresholds = column_mapping.get('female_thresholds', [])
        threshold_directions = column_mapping.get('threshold_directions', [])

        filtered_variables = []
        filtered_original_cols = []
        filtered_compared_cols = []
        filtered_male_thresholds = []
        filtered_female_thresholds = []
        filtered_threshold_directions = []

        for idx, (var_name, orig_col, comp_col) in enumerate(zip(variables, original_cols, compared_cols)):
            # Shared columns receive merge suffixes.
            orig_col_suffixed = f"{orig_col}_original"
            comp_col_suffixed = f"{comp_col}_compared"

            orig_exists = orig_col_suffixed in merged_df.columns
            comp_exists = comp_col_suffixed in merged_df.columns

            # Fall back to unsuffixed column names.
            if not orig_exists:
                if orig_col in merged_df.columns:
                    orig_col_suffixed = orig_col
                    orig_exists = True

            if not comp_exists:
                if comp_col in merged_df.columns:
                    comp_col_suffixed = comp_col
                    comp_exists = True

            if orig_exists and comp_exists:
                filtered_variables.append(var_name)
                filtered_original_cols.append(orig_col_suffixed)
                filtered_compared_cols.append(comp_col_suffixed)

                # Add threshold information if available
                if idx < len(male_thresholds):
                    filtered_male_thresholds.append(male_thresholds[idx])
                if idx < len(female_thresholds):
                    filtered_female_thresholds.append(female_thresholds[idx])
                if idx < len(threshold_directions):
                    filtered_threshold_directions.append(threshold_directions[idx])

                self.logger.info(f"✓ {var_name}: '{orig_col_suffixed}' <-> '{comp_col_suffixed}'")
            else:
                missing = []
                if not orig_exists:
                    missing.append(f"original '{orig_col_suffixed}'")
                if not comp_exists:
                    missing.append(f"compared '{comp_col_suffixed}'")
                self.logger.warning(f"✗ {var_name}: Missing {', '.join(missing)}")

        filtered_mapping = {
            'Variables': filtered_variables,
            'original_cols': filtered_original_cols,
            'compared_cols': filtered_compared_cols
        }

        # Add threshold information if it was present
        if filtered_male_thresholds:
            filtered_mapping['male_thresholds'] = filtered_male_thresholds
        if filtered_female_thresholds:
            filtered_mapping['female_thresholds'] = filtered_female_thresholds
        if filtered_threshold_directions:
            filtered_mapping['threshold_directions'] = filtered_threshold_directions

        self.logger.info(f"Filtered from {len(variables)} to {len(filtered_variables)} available pairs")
        return filtered_mapping

    def create_binary_labels_sex_stratified(self, values: np.ndarray, threshold: float, direction: str) -> np.ndarray:
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
                                 var_display_name: str = None,
                                 sex_label: str = None) -> Dict:
        """Evaluate train, validation, and test folds."""
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        has_test = (X_test is not None) and (y_test is not None)
        base_name = "fold" if var_display_name is None else "".join(
            c for c in var_display_name if c.isalnum() or c in (' ', '-', '_')
        ).rstrip().replace(' ', '_')
        sex_suffix = f"_{sex_label}" if sex_label else ""
        safe_name = f"{base_name}{sex_suffix}"

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
            train_auroc = roc_auc_score(y_train, y_train_proba)
            train_metrics['auroc'].append(train_auroc)
            train_metrics['auprc'].append(average_precision_score(y_train, y_train_proba))

            # Use the training-set Youden threshold.
            fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_proba)
            youden_idx_train = np.argmax(tpr_train - fpr_train)
            optimal_threshold_train = thresholds_train[youden_idx_train]
            y_train_pred = (y_train_proba >= optimal_threshold_train).astype(int)
            train_metrics['f1'].append(f1_score(y_train, y_train_pred))

            # TPR, TNR, FPR, FNR at optimal threshold
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

            # TPR, TNR, FPR, FNR at optimal threshold
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

            # Store fold results for median selection
            fold_result = {
                'fold_idx': fold_idx,
                'val_auroc': val_auroc,
                'test_auroc': test_auroc,
                'y_val_true': y_val,
                'y_val_proba': y_val_proba
            }

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

        # TPR, TNR, FPR, FNR at optimal threshold
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

            # TPR, TNR, FPR, FNR at optimal threshold
            tn_boot_opt, fp_boot_opt, fn_boot_opt, tp_boot_opt = confusion_matrix(y_boot, y_boot_pred).ravel()
            bootstrap_tpr.append(tp_boot_opt / (tp_boot_opt + fn_boot_opt) if (tp_boot_opt + fn_boot_opt) > 0 else 0.0)
            bootstrap_tnr.append(tn_boot_opt / (tn_boot_opt + fp_boot_opt) if (tn_boot_opt + fp_boot_opt) > 0 else 0.0)
            bootstrap_fpr.append(fp_boot_opt / (fp_boot_opt + tn_boot_opt) if (fp_boot_opt + tn_boot_opt) > 0 else 0.0)
            bootstrap_fnr.append(fn_boot_opt / (fn_boot_opt + tp_boot_opt) if (fn_boot_opt + tp_boot_opt) > 0 else 0.0)

            # DOR at optimal threshold
            if tp_boot_opt == 0 or tn_boot_opt == 0 or fp_boot_opt == 0 or fn_boot_opt == 0:
                tp_boot_adj = tp_boot_opt + 0.5
                tn_boot_adj = tn_boot_opt + 0.5
                fp_boot_adj = fp_boot_opt + 0.5
                fn_boot_adj = fn_boot_opt + 0.5
            else:
                tp_boot_adj, tn_boot_adj, fp_boot_adj, fn_boot_adj = tp_boot_opt, tn_boot_opt, fp_boot_opt, fn_boot_opt
            bootstrap_dor.append((tp_boot_adj * tn_boot_adj) / (fp_boot_adj * fn_boot_adj))

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
                               var_name: str, sex_label: str, optimal_threshold: float) -> None:
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

        # Create safe filename
        safe_name = "".join(c for c in var_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
        safe_sex = "".join(c for c in sex_label if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
        output_path = self.output_dir / f"predictions_{safe_name}_{safe_sex}.csv"

        df.to_csv(output_path, index=False, float_format='%.6f')
        self.logger.info(f"  Full dataset predictions saved: {output_path.name}")
        self.logger.info(f"  Total samples: {len(df)}")

    def export_confusion_matrix(self, full_results: Dict, y_true: np.ndarray,
                               var_name: str, sex_label: str, optimal_threshold: float) -> None:
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
        safe_sex = "".join(c for c in sex_label if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
        output_path = self.output_dir / f"confusion_matrix_{safe_name}_{safe_sex}.csv"
        df.to_csv(output_path, index=False)

        self.logger.info(f"  Confusion matrix saved: confusion_matrix_{safe_name}_{safe_sex}.csv")

    def export_cv_results_detailed_and_summary(self, cv_results: Dict, var_display_name: str,
                                               sex_label: str) -> None:
        """Export sex-specific fold metrics and summaries."""
        if 'test' not in cv_results:
            self.logger.info("  Skipping cv_results_detailed export (no external test)")
            return

        base_name = "".join(
            c for c in var_display_name if c.isalnum() or c in (' ', '-', '_')
        ).rstrip().replace(' ', '_')
        sex_suffix = f"_{sex_label}" if sex_label else ""
        safe_name = f"{base_name}{sex_suffix}"

        train_raw = cv_results['train_raw']
        val_raw = cv_results['val_raw']
        test_raw = cv_results['test']['raw']
        n_folds = len(train_raw['auroc'])
        metric_keys = ['auroc', 'auprc', 'f1', 'tpr', 'tnr', 'fpr', 'fnr', 'dor']

        rows = []
        for i in range(n_folds):
            row = {'fold': i}
            for m in metric_keys:
                row[f'train_{m}'] = train_raw[m][i]
                row[f'internal_val_{m}'] = val_raw[m][i]
                row[f'external_val_{m}'] = test_raw[m][i]
            row['train_auc'] = row['train_auroc']
            row['internal_val_auc'] = row['internal_val_auroc']
            row['external_val_auc'] = row['external_val_auroc']
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

        summary_row = {'n_folds': n_folds, 'variable': var_display_name, 'sex': sex_label}
        for m in metric_keys:
            summary_row[f'train_{m}_mean'] = float(np.mean(train_raw[m]))
            summary_row[f'train_{m}_std'] = float(np.std(train_raw[m], ddof=1))
            summary_row[f'internal_val_{m}_mean'] = float(np.mean(val_raw[m]))
            summary_row[f'internal_val_{m}_std'] = float(np.std(val_raw[m], ddof=1))
            summary_row[f'external_val_{m}_mean'] = float(np.mean(test_raw[m]))
            summary_row[f'external_val_{m}_std'] = float(np.std(test_raw[m], ddof=1))
        summary_path = self.output_dir / f"cv_results_summary_{safe_name}.csv"
        pd.DataFrame([summary_row]).to_csv(summary_path, index=False, float_format='%.6f')
        self.logger.info(f"  CV summary saved: {summary_path.name}")

    def plot_roc_curve(self, cv_results: Dict, full_results: Dict,
                      var_name: str, y_true: np.ndarray, sex_label: str, threshold: float) -> None:
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
        ax.set_title(f'ROC Curve: {var_name} - {sex_label} (Threshold ≥ {threshold})\n'
                    f'Train AUC: {cv_results["train"]["auroc_mean"]:.3f}±{cv_results["train"]["auroc_std"]:.3f}, '
                    f'Val AUC: {cv_results["val"]["auroc_mean"]:.3f}±{cv_results["val"]["auroc_std"]:.3f}',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        plt.tight_layout()
        safe_name = "".join(c for c in var_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
        sex_suffix = sex_label.lower()
        plt.savefig(self.output_dir / f"roc_curve_{safe_name}_{sex_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"  ROC curve saved: roc_curve_{safe_name}_{sex_suffix}.png")

    def plot_pr_curve(self, cv_results: Dict, full_results: Dict,
                     var_name: str, y_true: np.ndarray, sex_label: str, threshold: float) -> None:
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
        ax.set_title(f'Precision-Recall Curve: {var_name} - {sex_label} (Threshold ≥ {threshold})\n'
                    f'Train AUPRC: {cv_results["train"]["auprc_mean"]:.3f}±{cv_results["train"]["auprc_std"]:.3f}, '
                    f'Val AUPRC: {cv_results["val"]["auprc_mean"]:.3f}±{cv_results["val"]["auprc_std"]:.3f}',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        plt.tight_layout()
        safe_name = "".join(c for c in var_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
        sex_suffix = sex_label.lower()
        plt.savefig(self.output_dir / f"pr_curve_{safe_name}_{sex_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"  PR curve saved: pr_curve_{safe_name}_{sex_suffix}.png")

    def analyze_column_pairs(self, column_mapping: Dict[str, List[str]]) -> None:
        """Analyze multiple column pairs with sex-stratified analysis."""
        self.logger.info("-" * 40)
        self.logger.info("SEX-STRATIFIED COLUMN PAIR ANALYSIS")
        self.logger.info("-" * 40)

        # Merge data
        merged_df = self.analyze_id_overlap()

        # Optional: merge held-out test set
        test_merged_df = None
        test_sex_col_name = None
        test_id_col_name = None
        if self.has_test_set:
            test_merged_df = self.analyze_id_overlap_test()
            if f"{self.sex_col}_original" in test_merged_df.columns:
                test_sex_col_name = f"{self.sex_col}_original"
            elif self.sex_col in test_merged_df.columns:
                test_sex_col_name = self.sex_col
            else:
                self.logger.warning(f"Sex column '{self.sex_col}' not found in test merged data — disabling test eval")
                test_merged_df = None
            if test_merged_df is not None:
                if f"{self.original_id_col}_original" in test_merged_df.columns:
                    test_id_col_name = f"{self.original_id_col}_original"
                elif self.original_id_col in test_merged_df.columns:
                    test_id_col_name = self.original_id_col

        # Prefer the suffixed sex column.
        sex_col_name = None
        if f"{self.sex_col}_original" in merged_df.columns:
            sex_col_name = f"{self.sex_col}_original"
        elif self.sex_col in merged_df.columns:
            sex_col_name = self.sex_col
        else:
            self.logger.error(f"Sex column '{self.sex_col}' not found in merged data!")
            self.logger.info(f"Available columns: {list(merged_df.columns)}")
            raise ValueError(f"Sex column '{self.sex_col}' not found")

        self.logger.info(f"Using sex column: '{sex_col_name}'")

        # Validate sex values
        sex_values = merged_df[sex_col_name].dropna().unique()
        self.logger.info(f"Unique sex values found: {sex_values}")

        if self.male_value not in sex_values:
            self.logger.warning(f"Male value {self.male_value} not found in sex column!")
        if self.female_value not in sex_values:
            self.logger.warning(f"Female value {self.female_value} not found in sex column!")

        # Filter to only available columns
        filtered_mapping = self.filter_available_columns(column_mapping, merged_df)

        if len(filtered_mapping['Variables']) == 0:
            self.logger.error("No matching column pairs found!")
            return

        variables = filtered_mapping['Variables']
        original_cols = filtered_mapping['original_cols']
        compared_cols = filtered_mapping['compared_cols']
        male_thresholds = filtered_mapping.get('male_thresholds', [])
        female_thresholds = filtered_mapping.get('female_thresholds', [])
        threshold_directions = filtered_mapping.get('threshold_directions', [])

        self.logger.info(f"Total variable pairs to analyze: {len(variables)}")

        # Get ID column name
        id_col_name = None
        if f"{self.original_id_col}_original" in merged_df.columns:
            id_col_name = f"{self.original_id_col}_original"
        elif self.original_id_col in merged_df.columns:
            id_col_name = self.original_id_col

        if id_col_name:
            self.logger.info(f"Using ID column: '{id_col_name}'")
        else:
            self.logger.warning("Patient ID column not found - predictions will not include IDs")

        results_list = []

        for i, (var_name, orig_col, comp_col, male_threshold, female_threshold, direction) in enumerate(
            zip(variables, original_cols, compared_cols, male_thresholds, female_thresholds, threshold_directions)
        ):
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Processing {i+1}/{len(variables)}: {var_name}")
            self.logger.info(f"{'=' * 60}")
            self.logger.info(f"Original column: '{orig_col}'")
            self.logger.info(f"Compared column: '{comp_col}'")
            self.logger.info(f"Male threshold: {male_threshold}, Female threshold: {female_threshold}, Direction: {direction}")

            try:
                # Get data
                orig_data = merged_df[orig_col]
                comp_data = merged_df[comp_col]
                sex_data = merged_df[sex_col_name]
                id_data = merged_df[id_col_name] if id_col_name else None

                self.logger.info(f"Original data - Total: {len(orig_data)}, Missing: {orig_data.isna().sum()}")
                self.logger.info(f"Compared data - Total: {len(comp_data)}, Missing: {comp_data.isna().sum()}")
                self.logger.info(f"Sex data - Total: {len(sex_data)}, Missing: {sex_data.isna().sum()}")

                # Remove NaN values
                if id_data is not None:
                    mask = ~(orig_data.isna() | comp_data.isna() | sex_data.isna() | id_data.isna())
                    id_clean = id_data.loc[mask].values
                else:
                    mask = ~(orig_data.isna() | comp_data.isna() | sex_data.isna())
                    id_clean = None

                orig_clean = orig_data.loc[mask].values
                comp_clean = comp_data.loc[mask].values
                sex_clean = sex_data.loc[mask].values

                self.logger.info(f"Valid data after removing NaN: {len(orig_clean)}")

                # Split by sex
                male_mask = sex_clean == self.male_value
                female_mask = sex_clean == self.female_value

                n_males = np.sum(male_mask)
                n_females = np.sum(female_mask)

                self.logger.info(f"Male subjects: {n_males}")
                self.logger.info(f"Female subjects: {n_females}")

                # Analyze each sex separately
                for sex_label, sex_mask, threshold in [
                    ('Male', male_mask, male_threshold),
                    ('Female', female_mask, female_threshold)
                ]:
                    # Create display name with sex-specific threshold
                    var_display_name = f"{var_name}_{int(threshold)}"

                    direction_symbol = '<' if direction == 'less_than' else '>='
                    self.logger.info(f"\n--- Analyzing {sex_label} (threshold {direction_symbol} {threshold}) ---")

                    orig_sex = orig_clean[sex_mask]
                    comp_sex = comp_clean[sex_mask]
                    id_sex = id_clean[sex_mask] if id_clean is not None else None

                    self.logger.info(f"{sex_label} samples: {len(orig_sex)}")

                    if len(orig_sex) < 50:
                        self.logger.warning(f"Insufficient data for {var_name} - {sex_label}: {len(orig_sex)} samples")
                        continue

                    # Create binary labels with sex-specific threshold and direction
                    y = self.create_binary_labels_sex_stratified(orig_sex, threshold, direction)
                    X = comp_sex

                    # Check class distribution
                    n_diseased = np.sum(y == 1)
                    n_healthy = np.sum(y == 0)
                    self.logger.info(f"{sex_label}: Class distribution - Diseased: {n_diseased}, Healthy: {n_healthy}")

                    if n_diseased < 10 or n_healthy < 10:
                        self.logger.warning(f"Insufficient samples per class for {var_name} - {sex_label}")
                        continue

                    # Extract sex-filtered test slice
                    X_test_arr = None
                    y_test_arr = None
                    test_pid_arr = None
                    if test_merged_df is not None:
                        if orig_col in test_merged_df.columns and comp_col in test_merged_df.columns:
                            t_orig = test_merged_df[orig_col]
                            t_comp = test_merged_df[comp_col]
                            t_sex = test_merged_df[test_sex_col_name]
                            t_id = test_merged_df[test_id_col_name] if test_id_col_name is not None else None
                            if t_id is not None:
                                t_mask = ~(t_orig.isna() | t_comp.isna() | t_sex.isna() | t_id.isna())
                                t_id_clean = t_id.loc[t_mask].values
                            else:
                                t_mask = ~(t_orig.isna() | t_comp.isna() | t_sex.isna())
                                t_id_clean = None
                            t_orig_clean = t_orig.loc[t_mask].values
                            t_comp_clean = t_comp.loc[t_mask].values
                            t_sex_clean = t_sex.loc[t_mask].values
                            sex_val = self.male_value if sex_label == 'Male' else self.female_value
                            t_sex_mask = t_sex_clean == sex_val
                            y_test_arr = self.create_binary_labels_sex_stratified(
                                t_orig_clean[t_sex_mask], threshold, direction
                            )
                            X_test_arr = t_comp_clean[t_sex_mask]
                            test_pid_arr = t_id_clean[t_sex_mask] if t_id_clean is not None else None
                            self.logger.info(f"External test samples ({sex_label}): {len(y_test_arr)} "
                                             f"(pos={int(np.sum(y_test_arr == 1))}, neg={int(np.sum(y_test_arr == 0))})")
                        else:
                            self.logger.warning(f"Test merge missing columns for {var_display_name}; skipping test scoring")

                    # Perform 5-fold cross-validation within sex group
                    self.logger.info(f"Performing {self.cv_folds}-fold cross-validation for {sex_label}...")
                    cv_results = self.cross_validation_analysis(
                        X, y,
                        patient_ids=id_sex,
                        original_values=orig_sex,
                        compared_values=comp_sex,
                        X_test=X_test_arr,
                        y_test=y_test_arr,
                        test_patient_ids=test_pid_arr,
                        var_display_name=var_display_name,
                        sex_label=sex_label,
                    )

                    # Export per-fold detailed + summary when test set present
                    if 'test' in cv_results:
                        self.export_cv_results_detailed_and_summary(cv_results, var_display_name, sex_label)

                    if self.has_test_set:
                        self.logger.info(f"Skipping bootstrap CI on training merge for {sex_label} (external test provided)")
                        full_results = None
                    else:
                        # Calculate metrics with bootstrap CI on full dataset
                        self.logger.info(f"Calculating full dataset metrics with bootstrap CIs for {sex_label}...")
                        full_results = self.calculate_metrics_with_bootstrap_ci(
                            X, y,
                            patient_ids=id_sex,
                            original_values=orig_sex,
                            compared_values=comp_sex
                        )

                    # Export predictions only after bootstrap analysis.
                    if full_results is not None:
                        if id_sex is not None:
                            self.export_full_predictions(full_results, y, var_display_name, sex_label, full_results['f1_optimal_threshold'])
                        self.export_confusion_matrix(full_results, y, var_display_name, sex_label, full_results['f1_optimal_threshold'])

                    # Log results
                    self.logger.info("")
                    self.logger.info(f"Results for {var_name} - {sex_label}:")
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

                    # Plot only after bootstrap analysis.
                    if full_results is not None:
                        self.plot_roc_curve(cv_results, full_results, var_display_name, y, sex_label, threshold)
                        self.plot_pr_curve(cv_results, full_results, var_display_name, y, sex_label, threshold)

                    # Store results
                    results_list.append({
                        'variable': var_name,
                        'sex': sex_label,
                        'threshold': threshold,
                        'threshold_direction': direction,
                        'n_samples': len(orig_sex),
                        'n_diseased': n_diseased,
                        'n_healthy': n_healthy,
                        'cv_folds': self.cv_folds,
                        'cv_results': cv_results,
                        'full_results': full_results
                    })

            except Exception as e:
                self.logger.error(f"Error processing {var_name}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

        # Export summary CSV
        if len(results_list) > 0:
            self.export_summary_csv(results_list)

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("SEX-STRATIFIED PREDICTION ANALYSIS COMPLETED")
        self.logger.info("=" * 80)

    def export_summary_csv(self, results_list: List[Dict]) -> None:
        """Export summary CSV with all metrics including sex stratification."""
        self.logger.info("-" * 40)
        self.logger.info("EXPORTING SUMMARY CSV")
        self.logger.info("-" * 40)

        summary_data = []
        for res in results_list:
            cv = res['cv_results']
            full = res['full_results']

            row = {
                'Variable': res['variable'],
                'Sex': res['sex'],
                'threshold': res['threshold'],
                'threshold_direction': res['threshold_direction'],
                'n_samples': res['n_samples'],
                'n_diseased': res['n_diseased'],
                'n_healthy': res['n_healthy'],
                'cv_folds': res['cv_folds'],

                'AUROC_train': f"{cv['train']['auroc_mean']:.4f}, {cv['train']['auroc_std']:.4f}",
                'AUPRC_train': f"{cv['train']['auprc_mean']:.4f}, {cv['train']['auprc_std']:.4f}",
                'F1_train': f"{cv['train']['f1_mean']:.4f}, {cv['train']['f1_std']:.4f}",
                'DOR_train': f"{cv['train']['dor_mean']:.4f}, {cv['train']['dor_std']:.4f}",
                'TPR_train': f"{cv['train']['tpr_mean']:.4f}, {cv['train']['tpr_std']:.4f}",
                'TNR_train': f"{cv['train']['tnr_mean']:.4f}, {cv['train']['tnr_std']:.4f}",
                'FPR_train': f"{cv['train']['fpr_mean']:.4f}, {cv['train']['fpr_std']:.4f}",
                'FNR_train': f"{cv['train']['fnr_mean']:.4f}, {cv['train']['fnr_std']:.4f}",

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
                for m in ['auroc', 'auprc', 'f1', 'dor', 'tpr', 'tnr', 'fpr', 'fnr']:
                    row[f'{m.upper()}_external_test'] = (
                        f"{cv['test'][f'{m}_mean']:.4f}, {cv['test'][f'{m}_std']:.4f}"
                    )

            if full is not None:
                row['AUROC_full'] = f"{full['auroc']:.4f} [{full['auroc_ci_lower']:.4f}, {full['auroc_ci_upper']:.4f}]"
                row['AUPRC_full'] = f"{full['auprc']:.4f} [{full['auprc_ci_lower']:.4f}, {full['auprc_ci_upper']:.4f}]"
                row['F1_full'] = f"{full['f1']:.4f} [{full['f1_ci_lower']:.4f}, {full['f1_ci_upper']:.4f}]"
                row['F1_optimal_threshold'] = f"{full['f1_optimal_threshold']:.4f}"
                row['DOR_full'] = f"{full['dor']:.4f} [{full['dor_ci_lower']:.4f}, {full['dor_ci_upper']:.4f}]"
                row['TPR_full'] = f"{full['tpr']:.4f} [{full['tpr_ci_lower']:.4f}, {full['tpr_ci_upper']:.4f}]"
                row['TNR_full'] = f"{full['tnr']:.4f} [{full['tnr_ci_lower']:.4f}, {full['tnr_ci_upper']:.4f}]"
                row['FPR_full'] = f"{full['fpr']:.4f} [{full['fpr_ci_lower']:.4f}, {full['fpr_ci_upper']:.4f}]"
                row['FNR_full'] = f"{full['fnr']:.4f} [{full['fnr_ci_lower']:.4f}, {full['fnr_ci_upper']:.4f}]"

            summary_data.append(row)

        df = pd.DataFrame(summary_data)
        output_path = self.output_dir / "prediction_analysis_sex_stratified_summary.csv"
        df.to_csv(output_path, index=False)

        self.logger.info(f"Summary CSV saved: {output_path}")
        self.logger.info(f"  - Rows: {len(df)}")
        self.logger.info(f"  - Columns: {len(df.columns)}")


def main():
    parser = argparse.ArgumentParser(
        description='Sex-Stratified Prediction Analysis for Binary Classification Performance Evaluation',
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

    parser.add_argument('--sex_col', type=str, default='Sex',
                       help='Column name for sex information (default: Sex)')
    parser.add_argument('--male_value', type=int, default=1,
                       help='Value indicating male in sex column (default: 1)')
    parser.add_argument('--female_value', type=int, default=0,
                       help='Value indicating female in sex column (default: 0)')

    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for plots and results')

    # Single-variable CLI overrides.
    parser.add_argument('--variable', type=str, default=None,
                       help='Variable display name (single-variable override)')
    parser.add_argument('--original_col', type=str, default=None,
                       help='Original column name (single-variable override)')
    parser.add_argument('--compared_col', type=str, default=None,
                       help='Compared column name (single-variable override)')
    parser.add_argument('--male_threshold', type=float, default=None,
                       help='Male threshold (single-variable override)')
    parser.add_argument('--female_threshold', type=float, default=None,
                       help='Female threshold (single-variable override)')
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

    # Default sex-specific LVEDVi task.
    all_column_mapping = {
        'Variables': ['LVEDVi'],
        'original_cols': ['LVEDVi_ml/m2'],
        'compared_cols': ['LVEDVi_ml/m2'],
        'male_thresholds': [75],
        'female_thresholds': [62],
        'threshold_directions': ['greater_than'],
    }

    if (args.variable is not None and args.original_col is not None
            and args.compared_col is not None and args.male_threshold is not None
            and args.female_threshold is not None and args.direction is not None):
        all_column_mapping = {
            'Variables': [args.variable],
            'original_cols': [args.original_col],
            'compared_cols': [args.compared_col],
            'male_thresholds': [args.male_threshold],
            'female_thresholds': [args.female_threshold],
            'threshold_directions': [args.direction],
        }

    test_original = args.test_original_path if args.test_original_path else None
    test_compared = args.test_compared_path if args.test_compared_path else None

    # Initialize analyzer
    analyzer = VisionECGPhenotypeSexStratifiedAnalyzer(
        original_path=args.original_path,
        compared_path=args.compared_path,
        original_id_col=args.original_id_col,
        compared_id_col=args.compared_id_col,
        sex_col=args.sex_col,
        male_value=args.male_value,
        female_value=args.female_value,
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
