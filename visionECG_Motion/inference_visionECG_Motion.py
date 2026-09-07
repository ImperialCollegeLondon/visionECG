#!/usr/bin/env python3
import argparse
import logging
import os
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_visionECG_Motion import MotionDecoderConfig
from dataset_visionECG_Motion import FrameMeshDataset
from metrics_visionECG_Motion import compute_batch_metrics_per_sample
from model_visionECG_Motion import build_model_from_config
from vtk_generator_visionECG_Motion import VTKGenerator


def setup_logging(output_dir: str) -> logging.Logger:
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"inference_{timestamp}.log")

    logger = logging.getLogger("visionECG_Motion_inference")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def torch_load_checkpoint(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint(checkpoint_path: str, device, logger: logging.Logger) -> Dict:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    logger.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    if "epoch_num" in checkpoint:
        logger.info("Checkpoint epoch: %s", checkpoint["epoch_num"])
    if "val_loss" in checkpoint:
        logger.info("Validation loss: %s", checkpoint["val_loss"])
    return checkpoint


def initialize_model(checkpoint: Dict, device, logger: logging.Logger, demo_cols=None):
    saved_config = checkpoint.get("config") or {}
    demo_cols_effective = demo_cols or saved_config.get("demo_cols")
    if demo_cols_effective is None:
        raise ValueError(
            "demo_cols is required (pass --demo_cols; legacy checkpoints do not "
            "store demo_cols and must be supplied at inference time)"
        )

    config = MotionDecoderConfig(demo_cols=list(demo_cols_effective))
    for key, value in saved_config.items():
        if key in ("demo_cols", "demographic_dim"):
            continue
        if hasattr(config, key):
            setattr(config, key, value)
    if saved_config:
        logger.info("Using config stored in checkpoint")
    else:
        state_dict = checkpoint.get("state_dict", checkpoint)
        if "decoder.finallayer.0.weight" in state_dict:
            config.latent_dim = state_dict["decoder.finallayer.0.weight"].shape[1]
            config.residual_hidden_dim = config.latent_dim * 2
            logger.info("Auto-detected latent_dim=%s", config.latent_dim)
        else:
            logger.warning("No config in checkpoint and latent_dim could not be auto-detected")

    config.demographic_dim = len(config.demo_cols)

    model = build_model_from_config(config, verbose=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    logger.info("Model initialized")
    logger.info("  latent_dim: %s", config.latent_dim)
    logger.info("  seq_len: %s", config.seq_len)
    logger.info("  points: %s", config.points)
    logger.info("  demographic_dim: %s", config.demographic_dim)
    logger.info("  demo_cols: %s", config.demo_cols)
    logger.info("  con_emb: %s", config.con_emb)
    return model, config


def run_inference(
    checkpoint_path: str,
    memory_z_path: str,
    queries_path: str,
    patient_csv_path: str,
    output_dir: str = None,
    batch_size: int = 32,
    generate_vtk: bool = True,
    vtk_output_dir: str = None,
    compute_metrics: bool = True,
    gpu_id: int = 0,
    num_workers: int = 4,
    save_predictions: bool = False,
    demo_cols=None,
    target_seg_dir: str = None,
):
    if output_dir is None:
        checkpoint_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        output_dir = os.path.join(checkpoint_dir, "inference_results")
    os.makedirs(output_dir, exist_ok=True)

    if vtk_output_dir is None:
        vtk_output_dir = os.path.join(output_dir, "vtk")
    if generate_vtk:
        os.makedirs(vtk_output_dir, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("=" * 80)
    logger.info("visionECG_Motion inference")
    logger.info("=" * 80)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Memory z: %s", memory_z_path)
    logger.info("Queries: %s", queries_path)
    logger.info("Patient CSV: %s", patient_csv_path)
    logger.info("Output dir: %s", output_dir)
    logger.info("Conditioning: demographics -> z residual only")
    logger.info("Frame residuals: query latents only")

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, device, logger)
    model, config = initialize_model(checkpoint, device, logger, demo_cols=demo_cols)
    config.batch_size = batch_size
    config.num_workers = num_workers
    if target_seg_dir is not None:
        config.target_seg_dir = target_seg_dir

    logger.info("Creating inference dataset")
    dataset = FrameMeshDataset(
        memory_z_path=memory_z_path,
        queries_path=queries_path,
        patient_csv_path=patient_csv_path,
        target_seg_dir=config.target_seg_dir,
        config=config,
        is_train=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    vtk_generator = None
    if generate_vtk:
        vtk_generator = VTKGenerator(
            include_wall_thickness=True,
            include_segment_ids=True,
            default_wall_thickness=10.0,
            default_segment_id=1,
        )

    all_metrics = [] if compute_metrics else None
    all_predictions = [] if save_predictions else None
    all_eids = []
    total_frames_saved = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
            try:
                z = batch["z"].to(device)
                queries = batch["queries"].to(device)
                gt_meshes = batch["heart_v"].to(device)
                faces = batch["heart_f"]
                demographics = batch["demographics"].to(device)
                eids = batch["eid"]

                pred_meshes = model(z, queries, demographics)

                if save_predictions:
                    all_predictions.append(pred_meshes.cpu().numpy())
                all_eids.extend([int(eid) for eid in eids])

                if compute_metrics:
                    batch_metrics = compute_batch_metrics_per_sample(pred_meshes, gt_meshes)
                    for idx, eid in enumerate(eids):
                        batch_metrics[idx]["eid"] = int(eid)
                    all_metrics.extend(batch_metrics)

                if generate_vtk and vtk_generator is not None:
                    if faces.dim() == 3:
                        batch_faces = faces.cpu().numpy()
                    else:
                        batch_faces = faces[:, 0, :, :].cpu().numpy()
                    patient_ids = [str(int(eid)) for eid in eids]
                    saved_counts = vtk_generator.save_batch_sequences(
                        batch_vertices=pred_meshes.cpu(),
                        batch_faces=batch_faces,
                        batch_patient_ids=patient_ids,
                        output_base_dir=vtk_output_dir,
                        verbose=False,
                    )
                    total_frames_saved += sum(saved_counts)
            except Exception as exc:
                logger.error("Error processing batch %s: %s", batch_idx, exc)
                logger.exception(exc)
                continue

    results = {
        "num_samples": len(all_eids),
        "checkpoint_path": checkpoint_path,
        "output_dir": output_dir,
        "memory_z_path": memory_z_path,
        "queries_path": queries_path,
        "patient_csv_path": patient_csv_path,
    }

    if compute_metrics and all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(output_dir, "metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)

        summary_stats = {}
        for metric in ["hd", "hd90", "assd", "mse"]:
            if metric in metrics_df.columns:
                summary_stats[f"{metric}_mean"] = metrics_df[metric].mean()
                summary_stats[f"{metric}_std"] = metrics_df[metric].std()
                summary_stats[f"{metric}_median"] = metrics_df[metric].median()
                logger.info(
                    "%s: %.6f +/- %.6f",
                    metric.upper(),
                    summary_stats[f"{metric}_mean"],
                    summary_stats[f"{metric}_std"],
                )

        summary_path = os.path.join(output_dir, "summary.csv")
        pd.DataFrame([summary_stats]).to_csv(summary_path, index=False)
        results["metrics"] = summary_stats

    if save_predictions and all_predictions:
        predictions_array = np.concatenate(all_predictions, axis=0)
        predictions_path = os.path.join(output_dir, "predictions.npy")
        np.save(predictions_path, predictions_array)
        results["predictions_shape"] = predictions_array.shape

    if generate_vtk:
        results["vtk_output_dir"] = vtk_output_dir
        results["total_frames_saved"] = total_frames_saved

    logger.info("Inference complete. Results saved to %s", output_dir)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="visionECG_Motion inference")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--memory_z_path", type=str, required=True)
    parser.add_argument("--queries_path", type=str, required=True)
    parser.add_argument("--patient_csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--vtk_output_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--generate_vtk", action="store_true", default=True)
    parser.add_argument("--no_vtk", dest="generate_vtk", action="store_false")
    parser.add_argument("--compute_metrics", action="store_true", default=True)
    parser.add_argument("--no_metrics", dest="compute_metrics", action="store_false")
    parser.add_argument("--save_predictions", action="store_true", default=False)
    parser.add_argument("--demo_cols", type=str, nargs="+")
    parser.add_argument("--target_seg_dir", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        checkpoint_path=args.checkpoint_path,
        memory_z_path=args.memory_z_path,
        queries_path=args.queries_path,
        patient_csv_path=args.patient_csv_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        generate_vtk=args.generate_vtk,
        vtk_output_dir=args.vtk_output_dir,
        compute_metrics=args.compute_metrics,
        gpu_id=args.gpu_id,
        num_workers=args.num_workers,
        save_predictions=args.save_predictions,
        demo_cols=args.demo_cols,
        target_seg_dir=args.target_seg_dir,
    )
