#!/usr/bin/env python
# -*-coding:utf-8 -*-
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


N_ESTIMATORS_CEILING = 2000
EARLY_STOPPING_ROUNDS = 50


def apply_imbalance(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Apply one imbalance-correction strategy."""
    if strategy == 'smote_only':
        smote = SMOTE(random_state=seed)
        X2, y2 = smote.fit_resample(X, y)
        return X2, y2, None

    if strategy == 'spw_only':
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        if n_pos == 0:
            return X, y, None
        sw = np.where(y == 1, float(n_neg) / float(n_pos), 1.0).astype(np.float32)
        return X, y, sw

    if strategy == 'smote_spw':
        smote = SMOTE(random_state=seed)
        X2, y2 = smote.fit_resample(X, y)
        n_neg = int((y2 == 0).sum())
        n_pos = int((y2 == 1).sum())
        if n_pos == 0:
            return X2, y2, None
        sw = np.where(y2 == 1, float(n_neg) / float(n_pos), 1.0).astype(np.float32)
        return X2, y2, sw

    if strategy == 'none':
        return X, y, None

    raise ValueError(f"Unknown imbalance strategy: {strategy!r}")


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Binary-classification metrics used across all output CSVs."""
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.clip(np.asarray(y_pred_proba, dtype=np.float64), 0.0, 1.0)

    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true; AUC/AUPRC set to 0.")
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

    return {
        'loss': float(loss),
        'auc': float(auc),
        'auprc': float(auprc),
        'accuracy': float(accuracy),
        'f1': float(f1),
        'tpr': float(tpr),
        'tnr': float(tnr),
        'fpr': float(fpr),
        'fnr': float(fnr),
    }


def _suggest_params(trial: optuna.Trial, search_space: Dict[str, dict]) -> Dict:
    """Sample params from a caller-provided search-space dict."""
    params: Dict = {}
    for name, spec in search_space.items():
        ptype = spec['type']
        log = bool(spec.get('log', False))
        if ptype == 'int':
            params[name] = trial.suggest_int(name, int(spec['low']), int(spec['high']), log=log)
        elif ptype == 'float':
            params[name] = trial.suggest_float(name, float(spec['low']), float(spec['high']), log=log)
        elif ptype == 'categorical':
            params[name] = trial.suggest_categorical(name, list(spec['choices']))
        else:
            raise ValueError(f"Unknown search-space param type '{ptype}' for '{name}'")
    return params


def _make_xgb_classifier(
    params: Dict,
    use_gpu: bool,
    seed: int,
    n_estimators: int = N_ESTIMATORS_CEILING,
    use_early_stopping: bool = True,
) -> xgb.XGBClassifier:
    base = dict(params)
    base['n_estimators'] = int(n_estimators)
    kwargs = dict(
        objective='binary:logistic',
        eval_metric='auc',
        tree_method='hist',
        random_state=seed,
        n_jobs=int(os.environ.get('XGB_N_JOBS', 1)),
        verbosity=0,
    )
    if use_early_stopping:
        kwargs['early_stopping_rounds'] = EARLY_STOPPING_ROUNDS
    kwargs['device'] = 'cuda' if use_gpu else 'cpu'
    kwargs.update(base)
    return xgb.XGBClassifier(**kwargs)


def _objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    imbalance_strategy: str,
    use_gpu: bool,
    base_seed: int,
    n_splits: int,
    n_repeats: int,
    search_space: Dict[str, dict],
) -> float:
    """Return repeated-CV mean validation AUC."""
    params = _suggest_params(trial, search_space)

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=base_seed,
    )
    fold_aucs: List[float] = []
    best_iters: List[int] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train)):
        X_f_tr, y_f_tr = X_train[tr_idx], y_train[tr_idx]
        X_f_va, y_f_va = X_train[va_idx], y_train[va_idx]

        try:
            X_f_tr2, X_f_es, y_f_tr2, y_f_es = train_test_split(
                X_f_tr, y_f_tr, test_size=0.15,
                stratify=y_f_tr, random_state=base_seed + fold_idx,
            )
        except ValueError:
            X_f_tr2, X_f_es, y_f_tr2, y_f_es = train_test_split(
                X_f_tr, y_f_tr, test_size=0.15, random_state=base_seed + fold_idx,
            )

        X_fit, y_fit, sw = apply_imbalance(
            X_f_tr2, y_f_tr2, imbalance_strategy, seed=base_seed + fold_idx,
        )

        model = _make_xgb_classifier(
            params, use_gpu=use_gpu, seed=base_seed, use_early_stopping=True,
        )
        fit_kwargs = dict(X=X_fit, y=y_fit, eval_set=[(X_f_es, y_f_es)], verbose=False)
        if sw is not None:
            fit_kwargs['sample_weight'] = sw

        try:
            model.fit(**fit_kwargs)
        except Exception as e:
            logger.warning(f"Trial {trial.number} fold {fold_idx} failed: {e}. Pruning.")
            raise optuna.TrialPruned()

        va_proba = model.predict_proba(X_f_va)[:, 1]
        fold_auc = roc_auc_score(y_f_va, va_proba) if len(np.unique(y_f_va)) >= 2 else 0.0
        fold_aucs.append(fold_auc)

        bi = getattr(model, 'best_iteration', None)
        if bi is None:
            bi = model.n_estimators
        best_iters.append(int(bi))

        trial.report(float(np.mean(fold_aucs)), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr('best_iterations', best_iters)
    trial.set_user_attr('fold_aucs', fold_aucs)

    return float(np.mean(fold_aucs))


def optuna_search_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    imbalance_strategy: str,
    use_gpu: bool,
    n_trials: int,
    n_splits: int,
    n_repeats: int,
    seed: int,
    timeout: Optional[int],
    search_space: Dict[str, dict],
    logger_instance: Optional[logging.Logger] = None,
) -> Tuple[Dict, pd.DataFrame, optuna.Study, List[int]]:
    """Run TPE search with repeated cross-validation."""
    log = logger_instance or logger

    sampler = optuna.samplers.TPESampler(
        seed=seed, n_startup_trials=15, multivariate=True, group=True,
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)

    log.info(f"[Optuna] strategy={imbalance_strategy}, n_trials={n_trials}, "
             f"n_splits={n_splits}, n_repeats={n_repeats}, timeout={timeout}s, "
             f"search_space_keys={sorted(search_space.keys())}")

    def _wrapped_objective(trial: optuna.Trial) -> float:
        return _objective(
            trial, X_train, y_train,
            imbalance_strategy=imbalance_strategy,
            use_gpu=use_gpu, base_seed=seed,
            n_splits=n_splits, n_repeats=n_repeats,
            search_space=search_space,
        )

    study.optimize(
        _wrapped_objective, n_trials=n_trials, timeout=timeout,
        gc_after_trial=True, show_progress_bar=False,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    log.info(f"[Optuna] {len(completed)} complete / {len(pruned)} pruned / {len(failed)} failed")
    log.info(f"[Optuna] Best mean fold-val AUC: {study.best_value:.4f}")
    log.info(f"[Optuna] Best params: {study.best_params}")

    best_iters_at_best = study.best_trial.user_attrs.get('best_iterations', [])
    if best_iters_at_best:
        log.info(f"[Optuna] best trial per-fold best_iterations: {best_iters_at_best}")

    trials_df = study.trials_dataframe()
    return dict(study.best_params), trials_df, study, list(map(int, best_iters_at_best))


def _save_fold_predictions(
    checkpoint_dir: Optional[str],
    fold_idx: int,
    y_train_fold: np.ndarray, train_proba: np.ndarray, eids_train_fold: Optional[np.ndarray],
    y_internal_val: np.ndarray, iv_proba: np.ndarray, eids_internal_val: Optional[np.ndarray],
    y_external_test: np.ndarray, et_proba: np.ndarray, eids_external_test: Optional[np.ndarray],
) -> None:
    if checkpoint_dir is None:
        return

    def _emit(name, y_true, proba, eids):
        path = os.path.join(checkpoint_dir, f"fold_{fold_idx}_{name}.csv")
        if eids is None:
            eids = np.arange(len(y_true))
        pd.DataFrame({
            'eid': eids,
            'true_label': y_true.astype(int),
            'predicted_probability': np.asarray(proba, dtype=np.float64),
        }).to_csv(path, index=False)

    _emit('train', y_train_fold, train_proba, eids_train_fold)
    _emit('internal_val', y_internal_val, iv_proba, eids_internal_val)
    _emit('external_test', y_external_test, et_proba, eids_external_test)


def final_cv_with_best_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_internal_val: np.ndarray,
    y_internal_val: np.ndarray,
    X_external_test: np.ndarray,
    y_external_test: np.ndarray,
    eids_train: Optional[np.ndarray],
    eids_internal_val: Optional[np.ndarray],
    eids_external_test: Optional[np.ndarray],
    best_params: Dict,
    fixed_n_estimators: int,
    imbalance_strategy: str,
    use_gpu: bool = False,
    n_folds: int = 5,
    seed: int = 42,
    feature_names: Optional[List[str]] = None,
    checkpoint_dir: Optional[str] = None,
    logger_instance: Optional[logging.Logger] = None,
) -> Tuple[float, xgb.XGBClassifier, List[xgb.XGBClassifier], List[Dict],
           pd.DataFrame, pd.DataFrame, Dict]:
    """Refit `n_folds` models at `best_params` with FIXED n_estimators."""
    log = logger_instance or logger

    log.info("\n" + "="*80)
    log.info(f"FINAL {n_folds}-FOLD CV AT BEST PARAMS (strategy={imbalance_strategy})")
    log.info("="*80)
    log.info(f"Training samples: {len(X_train)}")
    log.info(f"Internal validation samples: {len(X_internal_val)}")
    log.info(f"External test samples: {len(X_external_test)}")
    log.info(f"Fixed n_estimators: {fixed_n_estimators}")
    log.info(f"Best params: {best_params}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_models: List[xgb.XGBClassifier] = []
    all_fold_indices: List[Dict[str, np.ndarray]] = []
    detailed_rows: List[Dict] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        log.info(f"\n--- Fold {fold_idx}/{n_folds - 1} ---")
        all_fold_indices.append({'train_idx': tr_idx, 'val_idx': va_idx})

        X_f_tr = X_train[tr_idx]; y_f_tr = y_train[tr_idx]
        X_f_va = X_train[va_idx]; y_f_va = y_train[va_idx]

        X_fit, y_fit, sw = apply_imbalance(
            X_f_tr, y_f_tr, imbalance_strategy, seed=seed + fold_idx,
        )

        model = _make_xgb_classifier(
            best_params, use_gpu=use_gpu, seed=seed,
            n_estimators=fixed_n_estimators, use_early_stopping=False,
        )
        fit_kwargs = dict(X=X_fit, y=y_fit, verbose=False)
        if sw is not None:
            fit_kwargs['sample_weight'] = sw
        model.fit(**fit_kwargs)

        train_proba = model.predict_proba(X_f_tr)[:, 1]
        iv_proba    = model.predict_proba(X_internal_val)[:, 1]
        et_proba    = model.predict_proba(X_external_test)[:, 1]
        va_proba    = model.predict_proba(X_f_va)[:, 1]

        train_m = compute_metrics(y_f_tr, train_proba)
        iv_m    = compute_metrics(y_internal_val, iv_proba)
        et_m    = compute_metrics(y_external_test, et_proba)
        va_m    = compute_metrics(y_f_va, va_proba)

        log.info(f"  AUC train={train_m['auc']:.4f} iv={iv_m['auc']:.4f} "
                 f"ext={et_m['auc']:.4f} cv_val={va_m['auc']:.4f}")

        row = {
            'alpha': 0,
            'fold': fold_idx,
            'best_iteration': int(fixed_n_estimators),
            'cv_val_fold_auc': va_m['auc'],
            'imbalance_strategy': imbalance_strategy,
        }
        for ds_name, ds_m in [('train', train_m), ('internal_val', iv_m), ('external_val', et_m)]:
            for k, v in ds_m.items():
                row[f"{ds_name}_{k}"] = v
        row['n_active_features'] = float(len(feature_names)) if feature_names is not None else 0.0
        detailed_rows.append(row)

        fold_models.append(model)

        eids_train_fold = eids_train[tr_idx] if eids_train is not None else None
        _save_fold_predictions(
            checkpoint_dir, fold_idx,
            y_f_tr, train_proba, eids_train_fold,
            y_internal_val, iv_proba, eids_internal_val,
            y_external_test, et_proba, eids_external_test,
        )

    cv_results_detailed = pd.DataFrame(detailed_rows)

    def _mean(col): return float(cv_results_detailed[col].mean()) if col in cv_results_detailed else np.nan
    def _std(col):  return float(cv_results_detailed[col].std())  if col in cv_results_detailed else np.nan

    summary_row = {
        'alpha': 0,
        'imbalance_strategy': imbalance_strategy,
        'fixed_n_estimators': int(fixed_n_estimators),
        'mean_internal_val_loss': _mean('internal_val_loss'),
        'std_internal_val_loss':  _std('internal_val_loss'),
        'mean_internal_val_auc':  _mean('internal_val_auc'),
        'std_internal_val_auc':   _std('internal_val_auc'),
        'mean_internal_val_f1':   _mean('internal_val_f1'),
        'std_internal_val_f1':    _std('internal_val_f1'),
        'mean_external_val_loss': _mean('external_val_loss'),
        'std_external_val_loss':  _std('external_val_loss'),
        'mean_external_val_auc':  _mean('external_val_auc'),
        'std_external_val_auc':   _std('external_val_auc'),
        'mean_external_val_f1':   _mean('external_val_f1'),
        'std_external_val_f1':    _std('external_val_f1'),
        'mean_n_active_features': _mean('n_active_features'),
    }
    cv_results_summary = pd.DataFrame([summary_row])

    log.info("\n" + "="*80)
    log.info("WITHIN-STRATEGY MEDIAN FOLD SELECTION (by internal_val_auc)")
    log.info("="*80)
    sorted_by_iv = cv_results_detailed.sort_values('internal_val_auc')
    sorted_folds = sorted_by_iv['fold'].values
    sorted_iv_aucs = sorted_by_iv['internal_val_auc'].values
    median_position = n_folds // 2
    median_fold_idx = int(sorted_folds[median_position])

    for i, (fi, auc_) in enumerate(zip(sorted_folds, sorted_iv_aucs)):
        marker = "  <- MEDIAN" if i == median_position else ""
        log.info(f"  Fold {int(fi)}: internal_val AUC={auc_:.4f}{marker}")

    median_row = cv_results_detailed[cv_results_detailed['fold'] == median_fold_idx].iloc[0].to_dict()
    median_fold_info = dict(median_row)
    median_fold_info['best_alpha'] = 0
    median_fold_info['median_fold_idx'] = median_fold_idx

    median_model = fold_models[median_fold_idx]

    return (0, median_model, fold_models, all_fold_indices,
            cv_results_detailed, cv_results_summary, median_fold_info)


def get_feature_importance_df(model: xgb.XGBClassifier, feature_names: List[str]) -> pd.DataFrame:
    """Gain-based feature importance with LASSO-compatible column names."""
    booster = model.get_booster()
    gain_dict = booster.get_score(importance_type='gain')
    rows = []
    for i, name in enumerate(feature_names):
        key = f"f{i}"
        gain = float(gain_dict.get(key, 0.0))
        rows.append({'feature': name, 'coefficient': gain, 'abs_coefficient': gain})
    df = pd.DataFrame(rows).sort_values('abs_coefficient', ascending=False).reset_index(drop=True)
    return df
