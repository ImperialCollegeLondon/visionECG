#!/usr/bin/env python

import logging
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, log_loss, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LassoClassifierWrapper:
    """Use clipped Lasso scores for binary classification."""

    def __init__(self, alpha: float = 1.0, random_state: int = 42):
        self.lasso = Lasso(alpha=alpha, max_iter=10000, random_state=random_state)
        self.alpha = alpha
        self.is_fitted = False

    def fit(self, X, y, sample_weight=None):
        self.lasso.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return np.clip(self.lasso.predict(X), 0, 1)

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_active_features(self, feature_names: List[str]):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting active features")
        active = [(n, c) for n, c in zip(feature_names, self.lasso.coef_) if abs(c) > 1e-10]
        active.sort(key=lambda x: abs(x[1]), reverse=True)
        return active

    def get_coefficients_df(self, feature_names: List[str]) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting coefficients")
        df = pd.DataFrame({'feature': feature_names, 'coefficient': self.lasso.coef_})
        df['abs_coefficient'] = df['coefficient'].abs()
        return df.sort_values('abs_coefficient', ascending=False)[['feature', 'coefficient', 'abs_coefficient']]

    def save(self, path: str):
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Saved model to {path}")

    @staticmethod
    def load(path: str) -> "LassoClassifierWrapper":
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Loaded model from {path}")
        return model


def compute_metrics(y_true, y_pred_proba, threshold: float = 0.5) -> Dict[str, float]:
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true; AUC/AUPRC undefined")
        auc = 0.0
        auprc = 0.0
        loss = 0.0
    else:
        y_pred_proba_clipped = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)
        loss = log_loss(y_true, y_pred_proba_clipped)
        auc = roc_auc_score(y_true, y_pred_proba)
        auprc = average_precision_score(y_true, y_pred_proba)

    y_pred = (y_pred_proba >= threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {'loss': loss, 'auc': auc, 'auprc': auprc, 'accuracy': accuracy, 'f1': f1,
            'tpr': tpr, 'tnr': tnr, 'fpr': fpr, 'fnr': fnr}


def _apply_imbalance(X, y, strategy: str, random_state: int):
    """Return (X_fit, y_fit, sample_weight) for the requested imbalance strategy."""
    if strategy == 'weight':
        sample_weight = compute_sample_weight('balanced', y)
        return X, y, sample_weight
    if strategy == 'smote':
        from imblearn.over_sampling import SMOTE
        X_res, y_res = SMOTE(random_state=random_state).fit_resample(X, y)
        return X_res, y_res, None
    if strategy == 'none':
        return X, y, None
    raise ValueError(f"Invalid imbalance strategy: {strategy}. Must be 'weight', 'smote' or 'none'.")


def cv_grid_search_with_external_val(
    X_train, y_train,
    X_val_ext, y_val_ext,
    alpha_grid,
    best_metric: str = 'loss',
    n_folds: int = 5,
    feature_names: Optional[List[str]] = None,
    imbalance: str = 'weight',
    random_state: int = 42,
) -> Tuple[float, LassoClassifierWrapper, List[LassoClassifierWrapper], List[Dict],
           pd.DataFrame, pd.DataFrame, Dict]:
    """Select alpha and median fold using internal validation only."""
    logger.info("\n5-FOLD CV GRID SEARCH")
    logger.info(f"Train N={len(X_train)}, external val N={len(X_val_ext)}, "
                f"folds={n_folds}, alphas={len(alpha_grid)}, imbalance={imbalance}, "
                f"best_metric={best_metric}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    detailed_results: List[Dict] = []
    all_models_by_alpha: Dict[float, List[LassoClassifierWrapper]] = {}
    all_fold_indices: List[Dict] = []

    for alpha_idx, alpha in enumerate(alpha_grid):
        fold_models: List[LassoClassifierWrapper] = []
        fold_internal_metrics: List[Dict] = []
        fold_external_metrics: List[Dict] = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            if alpha_idx == 0:
                all_fold_indices.append({'train_idx': train_idx, 'val_idx': val_idx})

            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_va, y_va = X_train[val_idx], y_train[val_idx]

            X_fit, y_fit, sw = _apply_imbalance(X_tr, y_tr, imbalance, random_state)

            model = LassoClassifierWrapper(alpha=alpha, random_state=random_state)
            model.fit(X_fit, y_fit, sample_weight=sw)

            train_metrics = compute_metrics(y_fit, model.predict_proba(X_fit))
            internal_metrics = compute_metrics(y_va, model.predict_proba(X_va))
            external_metrics = compute_metrics(y_val_ext, model.predict_proba(X_val_ext))

            n_active = len(model.get_active_features(feature_names)) if feature_names else 0

            row = {'alpha': alpha, 'fold': fold_idx, 'n_active_features': n_active}
            for prefix, m in (('train', train_metrics), ('internal_val', internal_metrics),
                              ('external_val', external_metrics)):
                for k, v in m.items():
                    row[f'{prefix}_{k}'] = v
            detailed_results.append(row)

            fold_models.append(model)
            fold_internal_metrics.append(internal_metrics)
            fold_external_metrics.append(external_metrics)

            logger.info(f"  alpha={alpha:.4f} fold={fold_idx}: "
                        f"internal AUC={internal_metrics['auc']:.3f}, "
                        f"external AUC={external_metrics['auc']:.3f}, active={n_active}")

        all_models_by_alpha[alpha] = fold_models

    cv_results_detailed = pd.DataFrame(detailed_results)

    summary_data = []
    for alpha in alpha_grid:
        rows = cv_results_detailed[cv_results_detailed['alpha'] == alpha]
        summary_data.append({
            'alpha': alpha,
            'mean_internal_val_loss': rows['internal_val_loss'].mean(),
            'std_internal_val_loss': rows['internal_val_loss'].std(),
            'mean_internal_val_auc': rows['internal_val_auc'].mean(),
            'std_internal_val_auc': rows['internal_val_auc'].std(),
            'mean_internal_val_f1': rows['internal_val_f1'].mean(),
            'std_internal_val_f1': rows['internal_val_f1'].std(),
            'mean_external_val_loss': rows['external_val_loss'].mean(),
            'std_external_val_loss': rows['external_val_loss'].std(),
            'mean_external_val_auc': rows['external_val_auc'].mean(),
            'std_external_val_auc': rows['external_val_auc'].std(),
            'mean_external_val_f1': rows['external_val_f1'].mean(),
            'std_external_val_f1': rows['external_val_f1'].std(),
            'mean_n_active_features': rows['n_active_features'].mean(),
        })
    cv_results_summary = pd.DataFrame(summary_data)

    if best_metric == 'auc':
        best_alpha_idx = cv_results_summary['mean_internal_val_auc'].idxmax()
    elif best_metric == 'loss':
        best_alpha_idx = cv_results_summary['mean_internal_val_loss'].idxmin()
    else:
        raise ValueError(f"Invalid best_metric: {best_metric}. Must be 'auc' or 'loss'.")
    best_alpha = float(cv_results_summary.loc[best_alpha_idx, 'alpha'])
    logger.info(f"Best alpha={best_alpha:.6f} (by internal {best_metric})")

    fold_models_at_best = all_models_by_alpha[best_alpha]
    best_alpha_results = cv_results_detailed[
        np.isclose(cv_results_detailed['alpha'].astype(float), best_alpha)
    ]

    # Select median fold using internal validation only.
    if best_metric == 'auc':
        sorted_rows = best_alpha_results.sort_values('internal_val_auc')
        metric_col = 'internal_val_auc'
    else:
        sorted_rows = best_alpha_results.sort_values('internal_val_loss')
        metric_col = 'internal_val_loss'
    sorted_folds = sorted_rows['fold'].values
    median_position = n_folds // 2
    median_fold_idx = int(sorted_folds[median_position])
    median_model = fold_models_at_best[median_fold_idx]

    logger.info(f"Fold ranking at alpha={best_alpha:.6f} by {metric_col}:")
    for i, (fold_idx, val) in enumerate(zip(sorted_folds, sorted_rows[metric_col].values)):
        marker = " <-- MEDIAN" if i == median_position else ""
        logger.info(f"  fold {int(fold_idx)}: {metric_col}={val:.4f}{marker}")

    median_fold_info = best_alpha_results[best_alpha_results['fold'] == median_fold_idx].iloc[0].to_dict()
    median_fold_info['best_alpha'] = best_alpha
    median_fold_info['median_fold_idx'] = median_fold_idx

    return (best_alpha, median_model, fold_models_at_best, all_fold_indices,
            cv_results_detailed, cv_results_summary, median_fold_info)
