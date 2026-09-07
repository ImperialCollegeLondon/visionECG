#!/usr/bin/env python3
"""Cox PH partial-LR tests for prognostic-index pairs."""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

DURATION_COL = "follow_up_days"
EVENT_COL = "event_status"
PROB_COL = "prob"

DEFAULT_MAX_WORKERS = 4
PLRT_DIRNAME = "plrt"
PLRT_SUMMARY_DIRNAME = "plrt_summary"


def format_landmark_dirname(landmark_years, seed):
    y = float(landmark_years)
    if abs(y - round(y)) < 1e-9:
        y_str = f"{int(round(y))}"
    else:
        y_str = f"{y:g}".replace(".", "p")
    return f"landmark_{y_str}yr_sd{int(seed)}"


def resolve_prefixed_path(abs_pre, path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return str(path)
    return str(Path(abs_pre).expanduser() / path)


def safe_slug(value):
    safe = str(value).strip().replace(" ", "_").replace(".", "_").replace("/", "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or "endpoint"


def apply_dirname_suffix(base_dirname: str, suffix: str) -> str:
    suffix = suffix.strip() if suffix else ""
    if suffix:
        return f"{base_dirname}_{suffix}"
    return base_dirname


def setup_job_logger(log_file):
    logger = logging.getLogger(f"job_{log_file.stem}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def load_pred_csv(path):
    df = pd.read_csv(path)
    if "eid" not in df.columns:
        raise ValueError(f"{path} missing required 'eid' column")
    df["eid"] = pd.to_numeric(df["eid"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["eid"]).copy()
    df["eid"] = df["eid"].astype(np.int64)
    return df


def build_merged_frame(ecg_csv: Path, pheno_csv: Path) -> pd.DataFrame:
    for p in (ecg_csv, pheno_csv):
        if not p.exists():
            raise FileNotFoundError(f"missing prediction CSV: {p}")

    ecg = load_pred_csv(ecg_csv)
    pheno = load_pred_csv(pheno_csv)
    if PROB_COL not in ecg.columns:
        raise ValueError(f"{ecg_csv} missing required '{PROB_COL}' column")
    if PROB_COL not in pheno.columns:
        raise ValueError(f"{pheno_csv} missing required '{PROB_COL}' column")

    ecg = ecg.rename(columns={PROB_COL: "prob_ecg"})
    pheno = pheno.rename(columns={PROB_COL: "prob_pheno"})

    ecg_keep = [c for c in ["eid", DURATION_COL, EVENT_COL, "prob_ecg"]
                if c in ecg.columns]
    pheno_keep = [c for c in ["eid", "prob_pheno"] if c in pheno.columns]

    merged = ecg[ecg_keep].merge(pheno[pheno_keep], on="eid", how="inner")
    merged = merged.dropna(subset=[DURATION_COL, EVENT_COL,
                                   "prob_ecg", "prob_pheno"]).copy()
    merged[DURATION_COL] = pd.to_numeric(merged[DURATION_COL], errors="coerce")
    merged[EVENT_COL] = pd.to_numeric(merged[EVENT_COL], errors="coerce")
    merged = merged[merged[DURATION_COL] > 0].copy()
    return merged.reset_index(drop=True)


def run_rscript(rscript_exe: str, r_script_path: Path, merged_csv: Path,
                output_json: Path, event: str, logger,
                pct_delta: float = 0.01) -> None:
    cmd = [
        rscript_exe, str(r_script_path),
        "--input", str(merged_csv),
        "--output", str(output_json),
        "--event", event,
        "--pct_delta", str(float(pct_delta)),
    ]
    logger.info(f"  Rscript cmd: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.stdout:
        for line in proc.stdout.rstrip().splitlines():
            logger.info(f"  [R stdout] {line}")
    if proc.stderr:
        for line in proc.stderr.rstrip().splitlines():
            logger.warning(f"  [R stderr] {line}")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Rscript exited with code {proc.returncode} for {event}. "
            "See log for stderr."
        )


def _flatten_ph_table(ph_obj) -> Dict[str, float]:
    """Flatten proportional-hazards diagnostics by term."""
    out: Dict[str, float] = {
        "ph_p_score_z_ecg": np.nan,
        "ph_p_score_z_pheno": np.nan,
        "ph_p_global": np.nan,
    }
    if ph_obj is None or (isinstance(ph_obj, float) and np.isnan(ph_obj)):
        return out
    if isinstance(ph_obj, list):
        pairs = (
            ((row.get("_row") or row.get("term")), row.get("p"))
            for row in ph_obj if isinstance(row, dict)
        )
    elif isinstance(ph_obj, dict) and "p" in ph_obj:
        row_names = ph_obj.get("_row", []) or ph_obj.get("term", [])
        pairs = zip(row_names, ph_obj.get("p", []))
    else:
        return out
    for name, pval in pairs:
        value = float(pval) if pval is not None else np.nan
        if name == "score_z_ecg":
            out["ph_p_score_z_ecg"] = value
        elif name == "score_z_pheno":
            out["ph_p_score_z_pheno"] = value
        elif name == "GLOBAL":
            out["ph_p_global"] = value
    return out


def flatten_plrt_json(payload: Dict, event: str) -> Dict:
    uni = payload.get("univariate", {})
    full = payload.get("full_model", {})
    nested = payload.get("nested_lrt", {})
    fine = payload.get("fine_2002", {})
    meta = payload.get("meta", {})
    scaling = payload.get("scaling", {})
    uni_ecg = uni.get("ecg", {})
    uni_pheno = uni.get("pheno", {})
    row = {
        "event": event,
        "n_test_cox": payload.get("n_test_cox"),
        "n_events":   payload.get("n_events"),
        "ci_level":   payload.get("ci_level"),
        "prob_mean_ecg":   scaling.get("mean_ecg"),
        "prob_sd_ecg":     scaling.get("sd_ecg"),
        "prob_mean_pheno": scaling.get("mean_pheno"),
        "prob_sd_pheno":   scaling.get("sd_pheno"),
        "pct_delta":       scaling.get("pct_delta"),
        "uni_ecg_hr":        uni_ecg.get("hr"),
        "uni_ecg_ci_low":    uni_ecg.get("ci_lo"),
        "uni_ecg_ci_high":   uni_ecg.get("ci_hi"),
        "uni_ecg_wald_p":    uni_ecg.get("wald_p"),
        "uni_ecg_c_index":         uni_ecg.get("c_index"),
        "uni_ecg_c_index_ci_low":  uni_ecg.get("c_index_ci_low"),
        "uni_ecg_c_index_ci_high": uni_ecg.get("c_index_ci_high"),
        "uni_ecg_hr_per_1pct":         uni_ecg.get("hr_per_1pct"),
        "uni_ecg_hr_per_1pct_ci_low":  uni_ecg.get("hr_per_1pct_ci_low"),
        "uni_ecg_hr_per_1pct_ci_high": uni_ecg.get("hr_per_1pct_ci_high"),
        "uni_ecg_beta_per_prob":       uni_ecg.get("beta_per_prob"),
        "uni_ecg_se_per_prob":         uni_ecg.get("se_per_prob"),
        "uni_pheno_hr":      uni_pheno.get("hr"),
        "uni_pheno_ci_low":  uni_pheno.get("ci_lo"),
        "uni_pheno_ci_high": uni_pheno.get("ci_hi"),
        "uni_pheno_wald_p":  uni_pheno.get("wald_p"),
        "uni_pheno_c_index":         uni_pheno.get("c_index"),
        "uni_pheno_c_index_ci_low":  uni_pheno.get("c_index_ci_low"),
        "uni_pheno_c_index_ci_high": uni_pheno.get("c_index_ci_high"),
        "uni_pheno_hr_per_1pct":         uni_pheno.get("hr_per_1pct"),
        "uni_pheno_hr_per_1pct_ci_low":  uni_pheno.get("hr_per_1pct_ci_low"),
        "uni_pheno_hr_per_1pct_ci_high": uni_pheno.get("hr_per_1pct_ci_high"),
        "uni_pheno_beta_per_prob":       uni_pheno.get("beta_per_prob"),
        "uni_pheno_se_per_prob":         uni_pheno.get("se_per_prob"),
        "full_ecg_hr":        full.get("ecg", {}).get("hr"),
        "full_ecg_ci_low":    full.get("ecg", {}).get("ci_lo"),
        "full_ecg_ci_high":   full.get("ecg", {}).get("ci_hi"),
        "full_ecg_wald_p":    full.get("ecg", {}).get("wald_p"),
        "full_ecg_hr_per_1pct":         full.get("ecg", {}).get("hr_per_1pct"),
        "full_ecg_hr_per_1pct_ci_low":  full.get("ecg", {}).get("hr_per_1pct_ci_low"),
        "full_ecg_hr_per_1pct_ci_high": full.get("ecg", {}).get("hr_per_1pct_ci_high"),
        "full_ecg_beta_per_prob":       full.get("ecg", {}).get("beta_per_prob"),
        "full_ecg_se_per_prob":         full.get("ecg", {}).get("se_per_prob"),
        "full_pheno_hr":      full.get("pheno", {}).get("hr"),
        "full_pheno_ci_low":  full.get("pheno", {}).get("ci_lo"),
        "full_pheno_ci_high": full.get("pheno", {}).get("ci_hi"),
        "full_pheno_wald_p":  full.get("pheno", {}).get("wald_p"),
        "full_pheno_hr_per_1pct":         full.get("pheno", {}).get("hr_per_1pct"),
        "full_pheno_hr_per_1pct_ci_low":  full.get("pheno", {}).get("hr_per_1pct_ci_low"),
        "full_pheno_hr_per_1pct_ci_high": full.get("pheno", {}).get("hr_per_1pct_ci_high"),
        "full_pheno_beta_per_prob":       full.get("pheno", {}).get("beta_per_prob"),
        "full_pheno_se_per_prob":         full.get("pheno", {}).get("se_per_prob"),
        "full_c_index":         full.get("concordance"),
        "full_c_index_var":     full.get("concordance_var"),
        "full_c_index_ci_low":  full.get("concordance_ci_low"),
        "full_c_index_ci_high": full.get("concordance_ci_high"),
        "full_log_likelihood": full.get("log_likelihood"),
        "lr_drop_ecg":    nested.get("drop_ecg", {}).get("stat"),
        "p_drop_ecg":     nested.get("drop_ecg", {}).get("p"),
        "lr_drop_pheno":  nested.get("drop_pheno", {}).get("stat"),
        "p_drop_pheno":   nested.get("drop_pheno", {}).get("p"),
        "fine_stat": fine.get("stat"),
        "fine_p":    fine.get("p"),
        "ll_ecg":    fine.get("ll_ecg"),
        "ll_pheno":  fine.get("ll_pheno"),
        "ll_diff":   fine.get("ll_diff"),
        "r_version": meta.get("r_version"),
        "survival_version": meta.get("survival_version"),
    }
    row.update(_flatten_ph_table(payload.get("ph_check")))
    return row


def process_event(job_params):
    (
        event, ecg_csv, pheno_csv, out_root, rscript_exe, r_script_path,
        pct_delta,
    ) = job_params

    out_dir = Path(out_root) / safe_slug(event) / PLRT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    slurm_suffix = f"_slurm{os.environ['SLURM_JOB_ID']}" if os.environ.get(
        "SLURM_JOB_ID"
    ) else ""
    log_file = out_dir / f"{safe_slug(event)}_plrt{slurm_suffix}.log"
    logger = setup_job_logger(log_file)
    logger.info(f"=== risk_stratification_analysis: {event} ===")
    logger.info(f"  ecg_csv  = {ecg_csv}")
    logger.info(f"  pheno_csv = {pheno_csv}")

    try:
        merged = build_merged_frame(Path(ecg_csv), Path(pheno_csv))
        merged_csv = out_dir / "merged_test.csv"
        merged.to_csv(merged_csv, index=False)
        logger.info(f"  merged frame: n={len(merged)} written to {merged_csv}")

        output_json = out_dir / "plrt_result.json"
        run_rscript(
            rscript_exe=rscript_exe,
            r_script_path=r_script_path,
            merged_csv=merged_csv,
            output_json=output_json,
            event=event,
            logger=logger,
            pct_delta=pct_delta,
        )

        with open(output_json) as f:
            payload = json.load(f)
        flat = flatten_plrt_json(payload, event)
        flat.update({
            "ecg_pred_csv":   str(Path(ecg_csv).resolve()),
            "pheno_pred_csv": str(Path(pheno_csv).resolve()),
        })
        return {"status": "success", "event": event, "row": flat,
                "out_dir": str(out_dir)}

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error(f"process_event failed for {event}: {error_msg}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "failed", "event": event,
                "out_dir": str(out_dir), "error": error_msg}


def write_summary(job_results, out_root):
    summary_dir = Path(out_root) / PLRT_SUMMARY_DIRNAME
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for res in job_results:
        if res.get("status") != "success":
            rows.append({
                "event": res.get("event"),
                "status": res.get("status"),
                "error": res.get("error"),
            })
            continue
        row = dict(res["row"])
        row["status"] = "success"
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = summary_dir / "plrt_summary.csv"
    df.to_csv(out_path, index=False)
    return out_path, df


def _build_jobs_from_cli(events, ecg_csvs, pheno_csvs) -> List[Dict]:
    if not (len(events) == len(ecg_csvs) == len(pheno_csvs)):
        raise ValueError(
            "--events, --ecg_csv, and --pheno_csv must have equal counts "
            f"(got {len(events)} / {len(ecg_csvs)} / {len(pheno_csvs)})"
        )
    return [
        {"event": ev, "ecg_pred_csv": ec, "pheno_pred_csv": pc}
        for ev, ec, pc in zip(events, ecg_csvs, pheno_csvs)
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cox PH partial-LR tests for prognostic-index pairs. Bash "
            "supplies per-event prediction-CSV paths (chosen manually as the "
            "median-fold-by-internal-val output from the XGB training runs)."
        )
    )
    parser.add_argument("--abs_pre", "--abs-pre", type=str, required=True,
                        help="Prefix applied to any relative --output_root.")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Parent dir; landmark subtree appended automatically.")
    parser.add_argument("--landmark_years", type=float, default=5.0)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--output_suffix", type=str, required=True,
                        help="Suffix appended to the landmark directory name.")
    parser.add_argument("--events", nargs="+", type=str, required=True,
                        help="Event names; matched by index to --ecg_csv/--pheno_csv.")
    parser.add_argument("--ecg_csv", nargs="+", type=str, required=True,
                        help="Per-event ECG prediction CSV (abs path).")
    parser.add_argument("--pheno_csv", nargs="+", type=str, required=True,
                        help="Per-event phenotype prediction CSV (abs path).")
    parser.add_argument("--rscript", type=str,
                        default=shutil.which("Rscript") or "Rscript",
                        help="Path to the Rscript executable.")
    parser.add_argument(
        "--r_script_path", type=str, default="",
        help="Path to risk_stratification_analysis.R. "
             "Default: alongside this Python file.",
    )
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help="Parallel workers across events.")
    parser.add_argument(
        "--pct_delta", type=float, default=0.01,
        help="Absolute-probability step for supplementary HR_per_1pct columns.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root = resolve_prefixed_path(args.abs_pre, args.output_root)

    r_script_path = Path(args.r_script_path) if args.r_script_path else (
        Path(__file__).resolve().parent / "risk_stratification_analysis.R"
    )
    if not r_script_path.exists():
        raise FileNotFoundError(
            f"risk_stratification_analysis.R not found: {r_script_path}"
        )
    rscript_exe = args.rscript
    if not shutil.which(rscript_exe) and not Path(rscript_exe).exists():
        raise FileNotFoundError(
            f"Rscript executable not found on PATH ({rscript_exe}). "
            "Activate the survival_r conda env or set --rscript."
        )

    base_dirname = format_landmark_dirname(args.landmark_years, args.random_seed)
    target_landmark_dirname = apply_dirname_suffix(base_dirname, args.output_suffix)
    run_root = Path(args.output_root) / target_landmark_dirname
    run_root.mkdir(parents=True, exist_ok=True)

    events_list = _build_jobs_from_cli(
        args.events, args.ecg_csv, args.pheno_csv,
    )

    with open(run_root / "plrt_run_config.json", "w") as f:
        json.dump({
            **vars(args),
            "resolved_target_landmark_dirname": target_landmark_dirname,
            "resolved_r_script_path": str(r_script_path),
            "resolved_events": events_list,
        }, f, indent=2, sort_keys=True, default=str)

    print(f"Output root : {run_root}")
    print(f"Events      : {len(events_list)}")
    print(f"Rscript     : {rscript_exe}")
    print(f"R script    : {r_script_path}")
    print(f"Max workers : {args.max_workers}")
    print(f"pct_delta   : {args.pct_delta}")

    jobs = [
        (item["event"], item["ecg_pred_csv"], item["pheno_pred_csv"],
         str(run_root), rscript_exe, r_script_path, float(args.pct_delta))
        for item in events_list
    ]
    total = len(jobs)

    if args.max_workers == 1:
        job_results = []
        for idx, job in enumerate(jobs, 1):
            print(f"[{idx}/{total}] {job[0]}")
            job_results.append(process_event(job))
    else:
        with Pool(processes=min(args.max_workers, total)) as pool:
            job_results = pool.map(process_event, jobs)

    out_path, _ = write_summary(job_results, str(run_root))

    successes = sum(1 for r in job_results if r.get("status") == "success")
    failures = total - successes
    print(f"Successful events: {successes}/{total}")
    print(f"Failed events    : {failures}/{total}")
    for r in job_results:
        if r.get("status") == "failed":
            print(f"  - {r.get('event')}: {r.get('error')}")
    print(f"Summary CSV -> {out_path}")
    if failures > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
