import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_visionECG_Flow import VisionECGFlowConfig
from model_visionECG_Flow import VisionECGFlowModel
from dataset_visionECG_Flow import (
    ECGMotionDataset_VisionECGFlow,
    collate_fn_visionECG_Flow,
)


# ECG ablation helpers
def _is_none_like_path(csv_path) -> bool:
    if csv_path is None:
        return True
    return isinstance(csv_path, str) and csv_path.strip().lower() in {'', 'none', 'null', 'na', 'n/a'}


def load_average_ecg_latent(csv_path, ecg_embed_dim: int, device):
    """Load the population-average ECG latent tensor."""
    if _is_none_like_path(csv_path):
        return None
    expected_cols = [f'ecg_embed_{idx}' for idx in range(ecg_embed_dim)]
    df = pd.read_csv(csv_path)
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Average ECG latent CSV missing {len(missing)} required columns "
            f"(need ecg_embed_0..ecg_embed_{ecg_embed_dim - 1}) in {csv_path}"
        )
    values = df.iloc[0][expected_cols].to_numpy(dtype=np.float32)
    return torch.from_numpy(values).to(device).unsqueeze(0)  # [1, D]


def _apply_ecg_ablation(ecg_embeddings, remove_ecg, avg_ecg_tensor):
    """Apply average or zero ECG ablation."""
    if not remove_ecg:
        return ecg_embeddings
    if avg_ecg_tensor is not None:
        return avg_ecg_tensor.expand_as(ecg_embeddings).clone()
    return torch.zeros_like(ecg_embeddings)


# Checkpoint loading
def load_checkpoint(checkpoint_path: str, device):
    """Load checkpoint written by main_visionECG_Flow.py."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config_dict = checkpoint['config']
    model_state_dict = checkpoint['model_state_dict']

    template = checkpoint.get('template')
    if template is not None:
        template = {k: v.to(device) for k, v in template.items()}

    motion_scaler = checkpoint.get('motion_scaler')

    return config_dict, model_state_dict, template, motion_scaler


# Sampling
def sample_x0(template_per_batch, batch_size, motion_embed_dim, device, alpha=1.0):
    if template_per_batch is None:
        return torch.randn(batch_size, motion_embed_dim, device=device)
    mean = template_per_batch['mean']
    std = template_per_batch['std']
    noise = torch.randn_like(mean)
    return mean + alpha * std * noise


@torch.no_grad()
def euler_sampler(model, demographics, ecg_embeddings, frame_indices,
                  template_per_batch, motion_embed_dim, device, n_steps):
    """Euler sampler."""
    model.eval()
    batch_size = demographics.shape[0]
    x_t = sample_x0(template_per_batch, batch_size, motion_embed_dim, device)
    h = 1.0 / n_steps
    for step in range(n_steps):
        t = step / n_steps
        t_tensor = torch.full((batch_size, 1), t, device=device)
        velocity = model(x_t, t_tensor, demographics, ecg_embeddings, frame_indices)
        x_t = x_t + h * velocity
    return x_t


def template_for_frame(template, frame_idx_value: int, batch_size: int,
                       config: VisionECGFlowConfig, device):
    """Per-batch template view for a given frame."""
    if template is None:
        return None
    if config.is_frame_resolved:
        mean = template['mean'][frame_idx_value - 1].to(device)
        std = template['std'][frame_idx_value - 1].to(device)
    else:
        mean = template['mean'].to(device)
        std = template['std'].to(device)
    return {
        'mean': mean.unsqueeze(0).expand(batch_size, -1).contiguous(),
        'std': std.unsqueeze(0).expand(batch_size, -1).contiguous(),
    }


# Inference loops
def generate_predictions_sequence_level(model, config, dataset, loader, template,
                                        device, motion_scaler, output_path,
                                        remove_ecg, avg_ecg_tensor,
                                        num_samples_per_case, n_steps):
    print(f"Inference (sequence_level) over {len(dataset)} samples — "
          f"num_samples_per_case={num_samples_per_case}, n_steps={n_steps}")

    model.eval()
    all_preds, all_eids = [], []

    with torch.no_grad():
        for batch_idx, (demographics, ecg_embeddings, _, _) in enumerate(
                tqdm(loader, desc="Generating")):
            demographics = demographics.to(device)
            ecg_embeddings = ecg_embeddings.to(device)
            ecg_embeddings = _apply_ecg_ablation(ecg_embeddings, remove_ecg, avg_ecg_tensor)

            batch_size = demographics.shape[0]
            frame_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
            template_b = template_for_frame(template, 0, batch_size, config, device)

            if num_samples_per_case <= 1:
                preds = euler_sampler(
                    model, demographics, ecg_embeddings, frame_indices,
                    template_b, config.motion_embed_dim, device, n_steps,
                )
            else:
                acc = torch.zeros(batch_size, config.motion_embed_dim, device=device)
                for _ in range(num_samples_per_case):
                    acc += euler_sampler(
                        model, demographics, ecg_embeddings, frame_indices,
                        template_b, config.motion_embed_dim, device, n_steps,
                    )
                preds = acc / num_samples_per_case

            preds_np = preds.cpu().numpy()
            if motion_scaler is not None:
                preds_np = motion_scaler.inverse_transform(preds_np)
            all_preds.append(preds_np)

            start = batch_idx * loader.batch_size
            end = min(start + batch_size, len(dataset))
            all_eids.extend(dataset.get_eids()[start:end])

    all_preds = np.concatenate(all_preds, axis=0)
    cols = ['eid_18545'] + [f'z_{i}' for i in range(1, config.motion_embed_dim + 1)]
    df = pd.DataFrame(np.column_stack([all_eids, all_preds]), columns=cols)
    try:
        df['eid_18545'] = df['eid_18545'].astype(int)
    except Exception:
        pass
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")
    return df


def generate_predictions_frame_resolved(model, config, dataset, loader, template,
                                        device, motion_scaler, output_path,
                                        remove_ecg, avg_ecg_tensor,
                                        num_samples_per_case, n_steps):
    print(f"Inference (frame_resolved) over {len(dataset)} samples — "
          f"num_samples_per_case={num_samples_per_case}, n_steps={n_steps}, "
          f"frames={config.num_frames}")

    model.eval()
    all_preds, all_eids = [], []

    with torch.no_grad():
        for batch_idx, (demographics, ecg_embeddings, _, _) in enumerate(
                tqdm(loader, desc="Batches")):
            demographics = demographics.to(device)
            ecg_embeddings = ecg_embeddings.to(device)
            ecg_embeddings = _apply_ecg_ablation(ecg_embeddings, remove_ecg, avg_ecg_tensor)

            batch_size = demographics.shape[0]
            preds_per_frame = torch.zeros(batch_size, config.num_frames,
                                          config.motion_embed_dim, device=device)

            for frame_idx in range(1, config.num_frames + 1):
                frame_indices = torch.full((batch_size,), frame_idx,
                                           dtype=torch.long, device=device)
                template_b = template_for_frame(template, frame_idx, batch_size, config, device)

                if num_samples_per_case <= 1:
                    p = euler_sampler(
                        model, demographics, ecg_embeddings, frame_indices,
                        template_b, config.motion_embed_dim, device, n_steps,
                    )
                else:
                    acc = torch.zeros(batch_size, config.motion_embed_dim, device=device)
                    for _ in range(num_samples_per_case):
                        acc += euler_sampler(
                            model, demographics, ecg_embeddings, frame_indices,
                            template_b, config.motion_embed_dim, device, n_steps,
                        )
                    p = acc / num_samples_per_case
                preds_per_frame[:, frame_idx - 1, :] = p

            preds_np = preds_per_frame.cpu().numpy()  # [B, F, D]
            if motion_scaler is not None:
                shape = preds_np.shape
                flat = preds_np.reshape(-1, shape[-1])
                flat = motion_scaler.inverse_transform(flat)
                preds_np = flat.reshape(shape)

            # Flatten frames into columns mesh_embed_{d}_t_{f}
            preds_flat = preds_np.reshape(batch_size, -1)  # [B, F*D]
            all_preds.append(preds_flat)

            start = batch_idx * loader.batch_size
            end = min(start + batch_size, len(dataset))
            all_eids.extend(dataset.get_eids()[start:end])

    all_preds = np.concatenate(all_preds, axis=0)
    cols = ['eid_18545']
    for f in range(1, config.num_frames + 1):
        for d in range(1, config.motion_embed_dim + 1):
            cols.append(f'mesh_embed_{d}_t_{f}')
    df = pd.DataFrame(np.column_stack([all_eids, all_preds]), columns=cols)
    try:
        df['eid_18545'] = df['eid_18545'].astype(int)
    except Exception:
        pass
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")
    return df


# CLI
def main():
    parser = argparse.ArgumentParser(
        description='Stochastic Flow-Matching ECG-to-Motion single-GPU inference'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .pt checkpoint produced by main_visionECG_Flow.py')
    parser.add_argument('--input_csv', type=str, required=True,
                        help='CSV with demographics + ECG (motion columns optional)')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_name', type=str, default='predictions.csv')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=1)

    parser.add_argument('--num_samples_per_case', type=int, default=100,
                        help='Number of stochastic rollouts averaged per case (default 100)')
    parser.add_argument('--num_inference_steps', type=int, default=100,
                        help='Number of Euler steps per rollout (default 100)')
    parser.add_argument('--remove_ecg', action='store_true',
                        help='Ablation: substitute ECG embeddings (mean if --ecg_latent_average is set, else zeros)')
    parser.add_argument('--ecg_latent_average', type=str, default='',
                        help='Population-average ECG latent CSV (columns ecg_embed_0..N-1). '
                             'Empty string forces zero-substitution when --remove_ecg is set.')

    args = parser.parse_args()

    # Device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Checkpoint
    config_dict, state_dict, template, motion_scaler = load_checkpoint(args.checkpoint, device)

    config = VisionECGFlowConfig()
    for k, v in config_dict.items():
        if hasattr(config, k):
            setattr(config, k, v)

    # CLI overrides
    config.batch_size = args.batch_size
    config.gpu_id = args.gpu_id
    config.device = str(device)

    print(f"Mode: {config.mode}")
    print(f"motion_embed_dim: {config.motion_embed_dim}")
    print(f"num_inference_steps: {args.num_inference_steps}")

    # Dataset
    dataset = ECGMotionDataset_VisionECGFlow(
        args.input_csv, config, motion_scaler=motion_scaler, is_train=False,
    )
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn_visionECG_Flow,
        pin_memory=(device.type == 'cuda'),
    )

    # Model
    model = VisionECGFlowModel(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # ECG ablation tensor
    avg_ecg_tensor = (
        load_average_ecg_latent(args.ecg_latent_average, config.ecg_embed_dim, device)
        if args.remove_ecg else None
    )

    # Inference
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)

    if config.is_frame_resolved:
        generate_predictions_frame_resolved(
            model, config, dataset, loader, template, device, motion_scaler,
            output_path,
            remove_ecg=args.remove_ecg,
            avg_ecg_tensor=avg_ecg_tensor,
            num_samples_per_case=args.num_samples_per_case,
            n_steps=args.num_inference_steps,
        )
    else:
        generate_predictions_sequence_level(
            model, config, dataset, loader, template, device, motion_scaler,
            output_path,
            remove_ecg=args.remove_ecg,
            avg_ecg_tensor=avg_ecg_tensor,
            num_samples_per_case=args.num_samples_per_case,
            n_steps=args.num_inference_steps,
        )

    print("Inference completed.")


if __name__ == "__main__":
    main()
