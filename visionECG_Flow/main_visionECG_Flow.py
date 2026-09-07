import argparse
import logging
import os
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_visionECG_Flow import VisionECGFlowConfig
from model_visionECG_Flow import VisionECGFlowModel
from dataset_visionECG_Flow import (
    ECGMotionDataset_VisionECGFlow,
    collate_fn_visionECG_Flow,
    compute_template_visionECG_Flow,
)


# Setup utilities
def setup_logging(config: VisionECGFlowConfig):
    log_file = os.path.join(config.log_dir, 'visionecg_flow_training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def _build_checkpoint(*, epoch, model, optimizer, config,
                      template, motion_scaler, val_loss,
                      train_losses, val_losses,
                      best_epoch, best_val_metric):
    """Checkpoint payload."""
    serialised_template = None
    if template is not None:
        serialised_template = {'mean': template['mean'].cpu(),
                               'std': template['std'].cpu()}
    return {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': config.to_dict(),
        'template': serialised_template,
        'motion_scaler': motion_scaler,
        'rng_state': {
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        },
        'train_losses': list(train_losses),
        'val_losses': list(val_losses),
        'best_epoch': best_epoch,
        'best_val_metric': best_val_metric,
    }


# Frame-index
def make_train_frame_indices(motion_targets: torch.Tensor,
                             config: VisionECGFlowConfig) -> torch.Tensor:
    batch_size = motion_targets.shape[0]
    device = motion_targets.device
    if config.is_frame_resolved:
        return torch.randint(1, config.num_frames + 1, (batch_size,),
                             dtype=torch.long, device=device)
    return torch.zeros(batch_size, dtype=torch.long, device=device)


def select_frame_targets(motion_targets: torch.Tensor,
                         frame_indices: torch.Tensor,
                         config: VisionECGFlowConfig) -> torch.Tensor:
    """Per-sample target latent."""
    if not config.is_frame_resolved:
        return motion_targets
    batch_size = motion_targets.shape[0]
    motion_dim = motion_targets.shape[2]
    out = torch.empty(batch_size, motion_dim, dtype=motion_targets.dtype,
                      device=motion_targets.device)
    for i in range(batch_size):
        out[i] = motion_targets[i, frame_indices[i] - 1]
    return out


def select_frame_template(template, frame_indices: torch.Tensor,
                          config: VisionECGFlowConfig, device) -> dict:
    """Per-batch template view."""
    if template is None:
        return None

    mean = template['mean'].to(device)
    std = template['std'].to(device)
    batch_size = frame_indices.shape[0]

    if config.is_frame_resolved:
        idx0 = (frame_indices - 1).clamp_min(0)
        mean_b = mean[idx0]                  # [B, D]
        std_b = std[idx0]                    # [B, D]
    else:
        mean_b = mean.unsqueeze(0).expand(batch_size, -1).contiguous()
        std_b = std.unsqueeze(0).expand(batch_size, -1).contiguous()

    return {'mean': mean_b, 'std': std_b}


def sample_x0(template_per_batch, batch_size: int, motion_embed_dim: int,
              device, alpha: float = 1.0) -> torch.Tensor:
    if template_per_batch is None:
        return torch.randn(batch_size, motion_embed_dim, device=device)
    mean = template_per_batch['mean']
    std = template_per_batch['std']
    noise = torch.randn_like(mean)
    return mean + alpha * std * noise


# Sampler
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


# Validation
def evaluate_model_visionECGFlow(model, val_loader, template, device, config,
                                 logger, dataset_name="Validation"):
    """Held-out velocity MSE."""
    model.eval()
    losses = []
    n_errors = 0

    with torch.no_grad():
        for batch_idx, (demographics, ecg_embeddings, motion_targets, info_b) in enumerate(val_loader):
            try:
                demographics = demographics.to(device)
                ecg_embeddings = ecg_embeddings.to(device)
                motion_targets = motion_targets.to(device)

                frame_indices = make_train_frame_indices(motion_targets, config)
                target_motion = select_frame_targets(motion_targets, frame_indices, config)
                template_b = select_frame_template(template, frame_indices, config, device)

                batch_size = target_motion.shape[0]
                t = torch.rand(batch_size, 1, device=device)

                x0 = sample_x0(template_b, batch_size, config.motion_embed_dim, device)
                x_t = t * target_motion + (1 - t) * x0
                target_velocity = target_motion - x0

                predicted_velocity = model(x_t, t, demographics, ecg_embeddings, frame_indices)
                loss = F.mse_loss(predicted_velocity, target_velocity)
                losses.append(loss.item())
            except Exception as exc:
                n_errors += 1
                logger.error(f"{dataset_name} batch {batch_idx} failed: {exc}")
                logger.debug("Traceback:", exc_info=True)

    if n_errors > 0:
        logger.warning(f"{dataset_name}: {n_errors} batches failed")

    val_loss = float(np.mean(losses)) if losses else float('inf')
    logger.info(f"{dataset_name} velocity MSE: {val_loss:.6f}")

    return {'velocity_mse': val_loss}


def train_single_config(config: VisionECGFlowConfig):
    logger = setup_logging(config)
    logger.info("Starting visionECG_Flow training")
    logger.info(f"Mode: {config.mode}")
    logger.info("Configuration:")
    for k, v in config.to_dict().items():
        if k.endswith("_path") or k.endswith("_dir"):
            v = "<hidden>" if v else "<not set>"
        logger.info(f"   {k}: {v}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device)

    # Datasets
    logger.info("Loading datasets...")
    train_dataset = ECGMotionDataset_VisionECGFlow(config.train_csv_path, config, is_train=True)

    val_dataset = None
    val_loader = None
    if config.val_csv_path:
        val_dataset = ECGMotionDataset_VisionECGFlow(
            config.val_csv_path, config,
            motion_scaler=train_dataset.get_motion_scaler(), is_train=False,
        )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=device.type == 'cuda',
        collate_fn=collate_fn_visionECG_Flow,
    )
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=device.type == 'cuda',
            collate_fn=collate_fn_visionECG_Flow,
        )

    if val_dataset is not None:
        logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    else:
        logger.info(f"Train: {len(train_dataset)} | Val: <not provided — skipping model selection>")

    # Template
    logger.info("Computing mean template")
    template = compute_template_visionECG_Flow(train_dataset)
    if template is not None:
        logger.info(f"Template mean shape: {tuple(template['mean'].shape)} | "
                    f"std shape: {tuple(template['std'].shape)}")

    # Model
    logger.info("Initialising VisionECGFlowModel...")
    model = VisionECGFlowModel(config).to(device)

    info = model.get_model_info()
    logger.info(f"Total parameters: {info['total_params']:,}")
    logger.info(f"Trainable parameters: {info['trainable_params']:,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay)

    # Training loop
    train_losses = []
    val_losses = []
    best_val_metric = float('inf')
    is_better = lambda new, old: new < old - config.min_delta
    patience_counter = 0
    best_epoch = 0

    logger.info("Starting training loop...")

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_losses = []
        total_loss = 0.0
        n_train_errors = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{config.num_epochs}')
        for batch_idx, (demographics, ecg_embeddings, motion_targets, info_b) in enumerate(pbar):
            try:
                optimizer.zero_grad()

                demographics = demographics.to(device)
                ecg_embeddings = ecg_embeddings.to(device)
                motion_targets = motion_targets.to(device)

                frame_indices = make_train_frame_indices(motion_targets, config)

                target_motion = select_frame_targets(motion_targets, frame_indices, config)
                template_b = select_frame_template(template, frame_indices, config, device)

                batch_size = target_motion.shape[0]
                t = torch.rand(batch_size, 1, device=device)

                x0 = sample_x0(template_b, batch_size, config.motion_embed_dim, device)
                x_t = t * target_motion + (1 - t) * x0
                target_velocity = target_motion - x0

                predicted_velocity = model(x_t, t, demographics, ecg_embeddings, frame_indices)
                velocity_loss = F.mse_loss(predicted_velocity, target_velocity)
                velocity_loss.backward()
                optimizer.step()

                total_loss += velocity_loss.item()
                epoch_losses.append(velocity_loss.item())
                pbar.set_postfix({
                    'Loss': f'{velocity_loss.item():.3f}',
                    'Avg_Loss': f'{total_loss / (batch_idx + 1):.3f}',
                })
            except Exception as exc:
                n_train_errors += 1
                logger.error(f"Train batch {batch_idx} (epoch {epoch}) failed: {exc}")
                logger.debug("Traceback:", exc_info=True)
                optimizer.zero_grad()
                continue

        if n_train_errors > 0:
            logger.warning(f"Epoch {epoch}: {n_train_errors} training batches failed")

        avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float('inf')
        train_losses.append(avg_train_loss)

        if val_loader is not None and (epoch % 1 == 0):
            metrics = evaluate_model_visionECGFlow(
                model, val_loader, template, device, config, logger, "Validation",
            )
            val_loss = metrics['velocity_mse']
            val_losses.append(val_loss)

            if is_better(val_loss, best_val_metric):
                best_val_metric = val_loss
                best_epoch = epoch
                patience_counter = 0

                checkpoint = _build_checkpoint(
                    epoch=epoch, model=model, optimizer=optimizer,
                    config=config, template=template,
                    motion_scaler=train_dataset.get_motion_scaler(),
                    val_loss=val_loss,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    best_epoch=best_epoch, best_val_metric=best_val_metric,
                )
                ckpt_path = os.path.join(config.output_dir, 'visionecg_flow_best_model.pt')
                torch.save(checkpoint, ckpt_path)
                logger.info(f"New best — checkpoint saved to {ckpt_path}")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{config.patience}")

            logger.info(
                f"[Best so far] epoch {best_epoch} | val loss: {best_val_metric:.6f}"
            )

            if patience_counter >= config.patience:
                logger.info(f"Early stopping triggered after {patience_counter} epochs")
                break

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch}: Training loss = {avg_train_loss:.6f}")

        if epoch % 20 == 0:
            checkpoint = _build_checkpoint(
                epoch=epoch, model=model, optimizer=optimizer,
                config=config, template=template,
                motion_scaler=train_dataset.get_motion_scaler(),
                val_loss=val_losses[-1] if val_losses else None,
                train_losses=train_losses,
                val_losses=val_losses,
                best_epoch=best_epoch, best_val_metric=best_val_metric,
            )
            ckpt_path = os.path.join(config.output_dir, f'visionecg_flow_epoch_{epoch:04d}.pt')
            torch.save(checkpoint, ckpt_path)
            logger.info(f"Periodic checkpoint saved to {ckpt_path}")

    logger.info("Training completed.")
    if best_epoch > 0:
        logger.info(f"Best epoch {best_epoch} | best val loss: {best_val_metric:.6f}")


# CLI
def main():
    config = VisionECGFlowConfig()

    parser = argparse.ArgumentParser(description='visionECG_Flow — unified sequence_level/frame_resolved FM')

    parser.add_argument('--mode', type=str, choices=['sequence_level', 'frame_resolved'])

    parser.add_argument('--gpu_id', type=int)
    parser.add_argument('--dim_hid', type=int)
    parser.add_argument('--con_emb', type=int)
    parser.add_argument('--ecg_emb', type=int)
    parser.add_argument('--num_blocks', type=int)
    parser.add_argument('--drop_rate', type=float)
    parser.add_argument('--t_emb', type=int)

    parser.add_argument('--num_frames', type=int)
    parser.add_argument('--frame_embed_dim', type=int)

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--run_id', type=str)
    parser.add_argument('--seed', type=int)

    parser.add_argument('--train_csv_path', type=str)
    parser.add_argument('--val_csv_path', type=str,
                        help='Validation CSV used for model selection; '
                             'omit to skip validation (no best_model.pt).')
    parser.add_argument('--base_output_dir', type=str)

    args = parser.parse_args()

    # Forward CLI overrides
    for attr in [
        'mode', 'gpu_id', 'dim_hid', 'con_emb', 'ecg_emb', 'num_blocks', 'drop_rate', 't_emb',
        'num_frames', 'frame_embed_dim', 'batch_size', 'learning_rate', 'weight_decay',
        'num_epochs', 'patience', 'num_workers', 'run_id',
        'seed', 'train_csv_path', 'val_csv_path', 'base_output_dir',
    ]:
        v = getattr(args, attr, None)
        if v is not None:
            setattr(config, attr, v)

    config.run_id = ''
    config.initialize_paths_and_device()

    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    train_single_config(config)


if __name__ == "__main__":
    main()
