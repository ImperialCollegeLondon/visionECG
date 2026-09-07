#!/usr/bin/env python3
"""Bootstrap task-split metrics using validation-derived Youden thresholds."""

import argparse
import os
import re
import sys
import warnings
from typing import Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats as scipy_stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


_N_JOBS = 1


MISSING_VALUE_LABEL = 'N/A'
DEFAULT_PROB_COL_PRIORITY = (
    'predicted_probability',
    'calibrated_probability',
    'raw_prediction_probability',
)

CANONICAL_TASKS = ('LVEF45', 'LVEF50', 'LVEDVi_Female62', 'LVEDVi_Male75', 'WT_Max13', 'WT_Max15')


def normalize_to_canonical_task(raw):
    """Map varied dataset labels onto CANONICAL_TASKS."""
    if raw is None:
        return None
    s = str(raw).strip()
    if s in CANONICAL_TASKS:
        return s
    lower = s.lower()
    if 'lvedvi' in lower:
        has_female = 'female' in lower
        has_male = 'male' in lower
        if has_female and '62' in s:
            return 'LVEDVi_Female62'
        if has_male and '75' in s:
            return 'LVEDVi_Male75'
    if 'lvef' in lower:
        if '45' in s:
            return 'LVEF45'
        if '50' in s:
            return 'LVEF50'
    if 'wt' in lower or 'wall_thickness' in lower:
        if '13' in s:
            return 'WT_Max13'
        if '15' in s:
            return 'WT_Max15'
    return s


def load_predictions(csv_path: str, prob_col: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load (y_true, y_prob) from a predictions CSV."""
    df = pd.read_csv(csv_path)
    if 'true_label' not in df.columns:
        raise ValueError(
            f"Missing 'true_label' column in {csv_path}. "
            f"Columns: {list(df.columns)}"
        )
    if prob_col and prob_col in df.columns:
        chosen = prob_col
    else:
        chosen = next((c for c in DEFAULT_PROB_COL_PRIORITY if c in df.columns), None)
        if chosen is None:
            raise ValueError(
                f"No probability column found in {csv_path}. "
                f"Requested: {prob_col!r}; tried: {DEFAULT_PROB_COL_PRIORITY}; "
                f"columns: {list(df.columns)}"
            )
        if prob_col and chosen != prob_col:
            print(f"  [warn] requested prob_col={prob_col!r} missing in {csv_path}; "
                  f"fell back to {chosen!r}")
    y = df['true_label'].values.astype(int)
    p = df[chosen].values.astype(float)
    if not np.all(np.isin(np.unique(y), [0, 1])):
        raise ValueError(f"Non-binary labels in {csv_path}: {np.unique(y)}")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError(
            f"Probabilities out of [0,1] in {csv_path}: "
            f"[{p.min():.4f}, {p.max():.4f}]"
        )
    return y, p


def find_optimal_threshold_youden(y: np.ndarray, p: np.ndarray) -> float:
    """Youden's J = max(TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y, p)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def _confusion_counts(y: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(((y_pred == 1) & (y == 1)).sum())
    tn = int(((y_pred == 0) & (y == 0)).sum())
    fp = int(((y_pred == 1) & (y == 0)).sum())
    fn = int(((y_pred == 0) & (y == 1)).sum())
    return tp, tn, fp, fn


def metrics_at_threshold(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    """Threshold-dependent metrics; DOR uses Haldane-0.5 continuity."""
    y_pred = (p >= thr).astype(int)
    tp, tn, fp, fn = _confusion_counts(y, y_pred)

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr_v = 1.0 - spec
    fnr_v = 1.0 - sens
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = (2 * ppv * sens / (ppv + sens)) if (ppv + sens) > 0 else 0.0

    if 0 in (tp, tn, fp, fn):
        a, b, c, d = tp + 0.5, fp + 0.5, fn + 0.5, tn + 0.5
    else:
        a, b, c, d = float(tp), float(fp), float(fn), float(tn)
    dor = (a * d) / (b * c)
    sens_c = a / (a + c)
    spec_c = d / (b + d)
    lr_plus = sens_c / (1.0 - spec_c) if (1.0 - spec_c) > 0 else np.inf
    lr_minus = (1.0 - sens_c) / spec_c if spec_c > 0 else np.inf

    return {
        'F1':           f1,
        'Sensitivity':  sens,
        'Specificity':  spec,
        'FPR':          fpr_v,
        'FNR':          fnr_v,
        'PPV':          ppv,
        'NPV':          npv,
        'LR_plus':      lr_plus,
        'LR_minus':     lr_minus,
        'DOR':          dor,
    }


def compute_auroc(y, p): return roc_auc_score(y, p)
def compute_auprc(y, p): return average_precision_score(y, p)
def compute_brier(y, p): return brier_score_loss(y, p)


def _percentile_ci_from_samples(samples: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lo, hi


def _safe_call(stat_fn, y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        v = stat_fn(y, p)
        return float(v) if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _boot_batch(y, p, stat_fn, seeds):
    out = np.empty(len(seeds), dtype=float)
    n = len(y)
    for k, s in enumerate(seeds):
        rng = np.random.default_rng(s)
        idx = rng.integers(0, n, size=n)
        out[k] = _safe_call(stat_fn, y[idx], p[idx])
    return out


def _jack_batch(y, p, stat_fn, indices):
    out = np.empty(len(indices), dtype=float)
    n = len(y)
    mask = np.ones(n, dtype=bool)
    for k, i in enumerate(indices):
        mask[i] = False
        out[k] = _safe_call(stat_fn, y[mask], p[mask])
        mask[i] = True
    return out


def _run_bootstrap_parallel(y, p, stat_fn, n_boot, base_seed):
    seed_seq = np.random.SeedSequence(base_seed).spawn(n_boot)
    seeds = np.array([s.generate_state(1)[0] for s in seed_seq], dtype=np.uint64)
    if _N_JOBS in (0, 1):
        return _boot_batch(y, p, stat_fn, seeds.tolist())
    n_chunks = max(1, min(abs(_N_JOBS) * 4, n_boot))
    chunks = np.array_split(seeds, n_chunks)
    parts = Parallel(n_jobs=_N_JOBS, backend='loky', verbose=0)(
        delayed(_boot_batch)(y, p, stat_fn, chunk.tolist()) for chunk in chunks if len(chunk)
    )
    return np.concatenate(parts) if parts else np.empty(0)


def _run_jackknife_parallel(y, p, stat_fn):
    n = len(y)
    all_idx = np.arange(n)
    if _N_JOBS in (0, 1):
        return _jack_batch(y, p, stat_fn, all_idx.tolist())
    n_chunks = max(1, min(abs(_N_JOBS) * 4, n))
    chunks = np.array_split(all_idx, n_chunks)
    parts = Parallel(n_jobs=_N_JOBS, backend='loky', verbose=0)(
        delayed(_jack_batch)(y, p, stat_fn, chunk.tolist()) for chunk in chunks if len(chunk)
    )
    return np.concatenate(parts) if parts else np.empty(0)


def _bca_ci_from_samples(point, boot_samples, jack_samples, ci_level):
    boot_valid = boot_samples[np.isfinite(boot_samples)]
    if len(boot_valid) < 10:
        return float('nan'), float('nan')
    frac_lt = float(np.mean(boot_valid < point))
    frac_lt = float(np.clip(frac_lt, 1e-6, 1 - 1e-6))
    z0 = scipy_stats.norm.ppf(frac_lt)
    jack_valid = jack_samples[np.isfinite(jack_samples)]
    if len(jack_valid) < 3:
        return _percentile_ci_from_samples(boot_valid, 1 - ci_level)
    jm = jack_valid.mean()
    num = np.sum((jm - jack_valid) ** 3)
    den = 6.0 * (np.sum((jm - jack_valid) ** 2)) ** 1.5
    a = num / den if den != 0 else 0.0

    alpha = 1 - ci_level
    z_lo = scipy_stats.norm.ppf(alpha / 2)
    z_hi = scipy_stats.norm.ppf(1 - alpha / 2)
    denom_lo = 1 - a * (z0 + z_lo)
    denom_hi = 1 - a * (z0 + z_hi)
    if denom_lo == 0 or denom_hi == 0:
        return _percentile_ci_from_samples(boot_valid, alpha)
    alpha_lo = scipy_stats.norm.cdf(z0 + (z0 + z_lo) / denom_lo)
    alpha_hi = scipy_stats.norm.cdf(z0 + (z0 + z_hi) / denom_hi)
    lo = float(np.percentile(boot_valid, 100 * alpha_lo))
    hi = float(np.percentile(boot_valid, 100 * alpha_hi))
    return lo, hi


def bca_ci_generic(
    y: np.ndarray,
    p: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    random_seed: int = 42,
    ci_level: float = 0.95,
) -> Tuple[float, float, float]:
    """BCa CI for a metric_fn(y, p)."""
    y = np.asarray(y)
    p = np.asarray(p)
    try:
        point = float(metric_fn(y, p))
    except Exception as e:
        warnings.warn(f"metric_fn failed on full sample: {e}")
        return float('nan'), float('nan'), float('nan')
    boot = _run_bootstrap_parallel(y, p, metric_fn, n_boot, random_seed)
    jack = _run_jackknife_parallel(y, p, metric_fn)
    lo, hi = _bca_ci_from_samples(point, boot, jack, ci_level)
    return point, lo, hi


def bca_ci_fixed_threshold(
    y: np.ndarray,
    p: np.ndarray,
    thr: float,
    metric_name: str,
    n_boot: int = 2000,
    random_seed: int = 42,
    ci_level: float = 0.95,
) -> Tuple[float, float, float]:
    """BCa CI at a fixed threshold."""
    def stat(y_r, p_r):
        return metrics_at_threshold(y_r, p_r, thr)[metric_name]
    return bca_ci_generic(y, p, stat, n_boot=n_boot, random_seed=random_seed, ci_level=ci_level)


def bca_ci_val_refit(
    y: np.ndarray,
    p: np.ndarray,
    metric_name: str,
    n_boot: int = 2000,
    random_seed: int = 42,
    ci_level: float = 0.95,
) -> Tuple[float, float, float]:
    """BCa CI on val with Youden refit per resample."""
    def stat(y_r, p_r):
        thr_r = find_optimal_threshold_youden(y_r, p_r)
        return metrics_at_threshold(y_r, p_r, thr_r)[metric_name]
    return bca_ci_generic(y, p, stat, n_boot=n_boot, random_seed=random_seed, ci_level=ci_level)


def bca_ci_log_dor(
    y: np.ndarray,
    p: np.ndarray,
    thr: float,
    n_boot: int = 2000,
    random_seed: int = 42,
    ci_level: float = 0.95,
) -> Tuple[float, float, float]:
    """BCa CI for DOR on log scale, exponentiated back."""
    def stat(y_r, p_r):
        d = metrics_at_threshold(y_r, p_r, thr)['DOR']
        if d <= 0 or not np.isfinite(d):
            return np.nan
        return np.log(d)
    _, log_lo, log_hi = bca_ci_generic(y, p, stat, n_boot=n_boot, random_seed=random_seed, ci_level=ci_level)
    dor_point = metrics_at_threshold(y, p, thr)['DOR']
    if not (np.isfinite(log_lo) and np.isfinite(log_hi)):
        return float(dor_point), float('nan'), float('nan')
    return float(dor_point), float(np.exp(log_lo)), float(np.exp(log_hi))


THRESHOLD_METRICS = ('F1', 'Sensitivity', 'Specificity', 'FPR', 'FNR',
                     'PPV', 'NPV', 'LR_plus', 'LR_minus')


def evaluate_split(
    y: np.ndarray,
    p: np.ndarray,
    thr_val: float,
    split_name: str,
    n_boot: int,
    random_seed: int,
) -> list:
    """Metric rows for one (dataset, split)."""
    rows = []
    is_val = (split_name == 'val')

    for metric_name, fn in (('AUROC', compute_auroc), ('AUPRC', compute_auprc), ('Brier', compute_brier)):
        val, lo, hi = bca_ci_generic(y, p, fn, n_boot=n_boot, random_seed=random_seed)
        rows.append({'metric': metric_name, 'value': val, 'ci_lower': lo, 'ci_upper': hi})

    for m in THRESHOLD_METRICS:
        if is_val:
            val, lo, hi = bca_ci_val_refit(y, p, m, n_boot=n_boot, random_seed=random_seed)
        else:
            val, lo, hi = bca_ci_fixed_threshold(y, p, thr_val, m, n_boot=n_boot, random_seed=random_seed)
        rows.append({'metric': m, 'value': val, 'ci_lower': lo, 'ci_upper': hi})

    if is_val:
        def stat_log_dor(y_r, p_r):
            try:
                thr_r = find_optimal_threshold_youden(y_r, p_r)
                d = metrics_at_threshold(y_r, p_r, thr_r)['DOR']
                if d <= 0 or not np.isfinite(d):
                    return np.nan
                return np.log(d)
            except Exception:
                return np.nan
        _, log_lo, log_hi = bca_ci_generic(y, p, stat_log_dor,
                                           n_boot=n_boot, random_seed=random_seed)
        dor_val = metrics_at_threshold(y, p, thr_val)['DOR']
        rows.append({
            'metric': 'DOR',
            'value': float(dor_val),
            'ci_lower': float(np.exp(log_lo)) if np.isfinite(log_lo) else float('nan'),
            'ci_upper': float(np.exp(log_hi)) if np.isfinite(log_hi) else float('nan'),
        })
    else:
        dor_val, dor_lo, dor_hi = bca_ci_log_dor(y, p, thr_val, n_boot=n_boot, random_seed=random_seed)
        rows.append({'metric': 'DOR', 'value': dor_val, 'ci_lower': dor_lo, 'ci_upper': dor_hi})

    return rows


def _plot_roc(y, p, out_path, auc_val, auc_lo, auc_hi, title_suffix=''):
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC (AUC = {auc_val:.3f} [{auc_lo:.3f}, {auc_hi:.3f}])')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC {title_suffix}')
    plt.legend(loc='lower right', fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out_path, dpi=200); plt.close()


def _plot_pr(y, p, out_path, ap_val, ap_lo, ap_hi, title_suffix=''):
    precision, recall, _ = precision_recall_curve(y, p)
    baseline = float(np.mean(y))
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR (AP = {ap_val:.3f} [{ap_lo:.3f}, {ap_hi:.3f}])')
    plt.plot([0, 1], [baseline, baseline], color='navy', lw=2, linestyle='--',
             label=f'Prevalence = {baseline:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f'PR {title_suffix}')
    plt.legend(loc='best', fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out_path, dpi=200); plt.close()


def _plot_cm(y, p, out_path, thr, title_suffix=''):
    y_pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=['Pred Neg', 'Pred Pos'],
           yticklabels=['True Neg', 'True Pos'],
           xlabel='Predicted', ylabel='True')
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            c = cm[i, j]
            pct = c / cm.sum() * 100
            ax.text(j, i, f'{c}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=12,
                    color='white' if cm[i, j] > thresh else 'black', fontweight='bold')
    ax.set_title(f'Confusion Matrix (thr={thr:.3f}) {title_suffix}')
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()


def _slug(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('_')


def _fmt_value_with_ci(value, lo, hi, digits=4):
    if not np.isfinite(value):
        return MISSING_VALUE_LABEL
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def process_model_dataset(
    model_name: str,
    dataset_label: str,
    train_path: str,
    val_path: str,
    test_path: str,
    prob_col: str,
    output_dir: str,
    n_boot: int,
    random_seed: int,
) -> list:
    """Evaluate all splits using validation-derived Youden thresholds."""
    canonical = normalize_to_canonical_task(dataset_label)
    if canonical != dataset_label:
        print(f"\n[{model_name} :: {dataset_label} -> {canonical}] processing...")
    else:
        print(f"\n[{model_name} :: {dataset_label}] processing...")
    dataset_label = canonical

    # Load val first: threshold is fit here.
    if not (isinstance(val_path, str) and val_path and os.path.isfile(val_path)):
        print(f"  [skip] val path missing/absent: {val_path!r} — emitting NaN placeholders")
        return _placeholder_rows(model_name, dataset_label)

    try:
        y_val, p_val = load_predictions(val_path, prob_col=prob_col)
    except Exception as e:
        print(f"  [skip] val load failed: {e}")
        return _placeholder_rows(model_name, dataset_label)

    try:
        thr_val = find_optimal_threshold_youden(y_val, p_val)
    except Exception as e:
        print(f"  [skip] Youden failed on val: {e}")
        return _placeholder_rows(model_name, dataset_label)
    print(f"  val threshold (Youden) = {thr_val:.4f}")

    splits = {'train': train_path, 'val': val_path, 'test': test_path}
    rows_out = []
    plot_dir_base = os.path.join(output_dir, _slug(model_name), _slug(dataset_label))
    os.makedirs(plot_dir_base, exist_ok=True)

    for split_name, path in splits.items():
        if not (isinstance(path, str) and path and os.path.isfile(path)):
            print(f"  [{split_name}] path missing/absent — placeholders")
            rows_out.extend(_placeholder_split_rows(model_name, dataset_label, split_name, thr_val))
            continue
        try:
            if split_name == 'val':
                y, p = y_val, p_val
            else:
                y, p = load_predictions(path, prob_col=prob_col)
        except Exception as e:
            print(f"  [{split_name}] load failed: {e}")
            rows_out.extend(_placeholder_split_rows(model_name, dataset_label, split_name, thr_val))
            continue

        n_samples = int(len(y))
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        print(f"  [{split_name}] N={n_samples} (pos={n_pos}, neg={n_neg}) — bootstrapping...")

        metric_rows = evaluate_split(y, p, thr_val, split_name, n_boot=n_boot, random_seed=random_seed)

        auc_row = next(r for r in metric_rows if r['metric'] == 'AUROC')
        ap_row  = next(r for r in metric_rows if r['metric'] == 'AUPRC')
        _plot_roc(y, p, os.path.join(plot_dir_base, f'{split_name}_roc.png'),
                  auc_row['value'], auc_row['ci_lower'], auc_row['ci_upper'],
                  f'{model_name} / {dataset_label} / {split_name}')
        _plot_pr(y, p, os.path.join(plot_dir_base, f'{split_name}_pr.png'),
                 ap_row['value'], ap_row['ci_lower'], ap_row['ci_upper'],
                 f'{model_name} / {dataset_label} / {split_name}')
        _plot_cm(y, p, os.path.join(plot_dir_base, f'{split_name}_cm.png'),
                 thr_val, f'{model_name} / {dataset_label} / {split_name}')

        for r in metric_rows:
            rows_out.append({
                'model': model_name,
                'dataset': dataset_label,
                'split': split_name,
                'metric': r['metric'],
                'value': r['value'],
                'ci_lower': r['ci_lower'],
                'ci_upper': r['ci_upper'],
                'value_with_ci': _fmt_value_with_ci(r['value'], r['ci_lower'], r['ci_upper']),
                'n_samples': n_samples,
                'n_diseased': n_pos,
                'n_healthy': n_neg,
                'optimal_threshold': thr_val,
            })

    return rows_out


def _all_metric_names() -> list:
    return ['AUROC', 'AUPRC', 'Brier'] + list(THRESHOLD_METRICS) + ['DOR']


def _placeholder_rows(model_name: str, dataset_label: str) -> list:
    rows = []
    for split_name in ('train', 'val', 'test'):
        rows.extend(_placeholder_split_rows(model_name, dataset_label, split_name, float('nan')))
    return rows


def _placeholder_split_rows(model_name, dataset_label, split_name, thr):
    return [{
        'model': model_name,
        'dataset': dataset_label,
        'split': split_name,
        'metric': m,
        'value': float('nan'),
        'ci_lower': float('nan'),
        'ci_upper': float('nan'),
        'value_with_ci': MISSING_VALUE_LABEL,
        'n_samples': 0,
        'n_diseased': 0,
        'n_healthy': 0,
        'optimal_threshold': thr,
    } for m in _all_metric_names()]


def load_input_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'dataset_label', 'train_path', 'val_path', 'test_path'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description='BCa bootstrap analysis for binary disease prediction.',
    )
    parser.add_argument('--input_tables', nargs='+', type=str, required=True,
                        help='Per-model path tables (dataset_label, train_path, val_path, test_path).')
    parser.add_argument('--model_names', nargs='+', type=str, required=True,
                        help='Display names aligned with --input_tables.')
    parser.add_argument('--prob_cols', nargs='+', type=str, required=True,
                        help='Probability column names aligned with --input_tables.')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--n_bootstraps', type=int, default=1000)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--n_jobs', type=int, default=96,
                        help='Parallel workers per BCa call; -1 uses all cores.')
    args = parser.parse_args()

    if not (len(args.input_tables) == len(args.model_names) == len(args.prob_cols)):
        parser.error(
            f"--input_tables ({len(args.input_tables)}), "
            f"--model_names ({len(args.model_names)}), "
            f"--prob_cols ({len(args.prob_cols)}) must have the same length."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    global _N_JOBS
    _N_JOBS = args.n_jobs

    print('=' * 80)
    print('Multi-split BCa bootstrap analysis')
    print('=' * 80)
    print(f'Output directory:  {args.output_dir}')
    print(f'Bootstrap samples: {args.n_bootstraps}')
    print(f'Parallel workers:  {args.n_jobs}')
    print(f'Random seed:       {args.random_seed}')
    print(f'Models:            {args.model_names}')

    all_rows = []
    for table_path, model_name, prob_col in zip(args.input_tables, args.model_names, args.prob_cols):
        print(f"\n{'=' * 80}\nModel: {model_name}\n  table: {table_path}\n  prob_col: {prob_col}\n{'=' * 80}")
        try:
            tbl = load_input_table(table_path)
        except Exception as e:
            print(f"  [skip model] {e}")
            continue

        for _, row in tbl.iterrows():
            all_rows.extend(process_model_dataset(
                model_name=model_name,
                dataset_label=str(row['dataset_label']),
                train_path=str(row['train_path']) if pd.notna(row['train_path']) else '',
                val_path=str(row['val_path']) if pd.notna(row['val_path']) else '',
                test_path=str(row['test_path']) if pd.notna(row['test_path']) else '',
                prob_col=prob_col,
                output_dir=args.output_dir,
                n_boot=args.n_bootstraps,
                random_seed=args.random_seed,
            ))

    if not all_rows:
        print('No rows produced — nothing to write.')
        sys.exit(1)

    out_df = pd.DataFrame(all_rows, columns=[
        'model', 'dataset', 'split', 'metric', 'value', 'ci_lower', 'ci_upper',
        'value_with_ci', 'n_samples', 'n_diseased', 'n_healthy', 'optimal_threshold',
    ])
    out_path = os.path.join(args.output_dir, 'metrics_summary_all_datasets.csv')
    out_df.to_csv(out_path, index=False, float_format='%.6f')
    print(f"\nWrote {out_path} ({len(out_df)} rows)")

    # Split summary
    for split in ('train', 'val', 'test'):
        n = int((out_df['split'] == split).sum())
        print(f"  split={split:5s}: {n} rows")


if __name__ == '__main__':
    main()
