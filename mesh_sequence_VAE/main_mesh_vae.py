import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import datetime
import logging
import time

from data.dataloader_mesh_vae import MeshVAEDataset
import loss.loss as Loss
from config_mesh_vae import load_config
import timeit
from torch.utils.tensorboard import SummaryWriter
import model.mesh_vae as mesh_vae_mod

from loss.metrics import (
    compute_mesh_metrics_for_sequence,
    EpochMetricsAccumulator,
    print_metrics_summary,
)
from util.utils import load_checkpoint_and_setup_paths, setup_training_directories


class WarmupScheduler:
    """Linear LR warmup over a fixed batch budget."""

    def __init__(self, optimizer, warmup_batches, start_lr, target_lr):
        self.optimizer = optimizer
        self.warmup_batches = warmup_batches
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.current_batch = 0

        initial_lr = target_lr if warmup_batches == 0 else start_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = initial_lr

    def step(self):
        if self.current_batch < self.warmup_batches:
            lr = self.start_lr + (self.target_lr - self.start_lr) * (self.current_batch / self.warmup_batches)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.current_batch += 1
            return True
        return False

    def is_warmup_complete(self):
        return self.current_batch >= self.warmup_batches


def setup_logger(cp_path, config):
    """Return a logger writing to console and training.log."""
    logger = logging.getLogger('MeshVAETraining')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    log_file = os.path.join(cp_path, 'training.log')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                                datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    logger.info('=' * 80)
    logger.info('Configuration Parameters:')
    for attr, value in config.__dict__.items():
        logger.info(f"   {attr}: {value}")
    logger.info('=' * 80)

    return logger


def save_model(mesh_vae, optimizer, epoch, train_loss, val_loss, checkpoint_name, model_type="best", logger=None):
    """Save checkpoint with training metadata."""
    checkpoint = {
        'state_dict': mesh_vae.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch_num': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_type': model_type,
        'save_time': datetime.datetime.now().isoformat(),
    }
    torch.save(checkpoint, checkpoint_name)

    msg1 = f"{model_type.capitalize()} model saved: {checkpoint_name}"
    msg2 = f"   Epoch: {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
    if logger:
        logger.info(msg1)
        logger.info(msg2)
    else:
        print(msg1)
        print(msg2)


def compute_batch_mesh_metrics(v_out, v_gt, patient_ids, logger=None):
    """Return batch Hausdorff and surface distances."""
    batch_size = v_out.shape[0]
    hd_values = []
    assd_values = []

    for b in range(batch_size):
        try:
            avg_hd, avg_assd = compute_mesh_metrics_for_sequence(v_out[b], v_gt[b])
            hd_values.append(avg_hd.item())
            assd_values.append(avg_assd.item())
        except Exception as e:
            msg = f"Warning: metrics failed for patient {patient_ids[b] if patient_ids else b}: {e}"
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            continue

    return hd_values, assd_values


def train(model, trainloader, optimizer, device, config, writer, epoch, warmup_scheduler, logger=None):
    """One training epoch with gradient accumulation."""
    model.train()
    avg_loss = []
    avg_recon_loss = []
    batch_losses = []

    metrics_accumulator = EpochMetricsAccumulator()

    data_iter = iter(trainloader)
    num_effective_batches = len(trainloader) // config.accumulation_steps

    pbar = tqdm(range(num_effective_batches), desc=f"Epoch {epoch}", leave=False)
    for effective_idx in pbar:
        try:
            optimizer.zero_grad()
            effective_batch_loss = 0.0
            effective_batch_recon_loss = 0.0

            for _ in range(config.accumulation_steps):
                try:
                    data = next(data_iter)
                    heart_v, heart_f, heart_e, _ = data

                    heart_v = heart_v.to(device)
                    heart_f = heart_f.to(device)
                    heart_e = heart_e.to(device)

                    v_out, logvar, mu = model(heart_v, heart_f, heart_e)

                    loss, loss_recon = Loss.VAECELoss(v_out, heart_v, heart_f, logvar, mu,
                                                     beta=config.beta, lambd=config.lambd,
                                                     lambd_s=config.lambd_s, loss=config.loss)

                    loss = loss / config.accumulation_steps
                    loss_recon = loss_recon / config.accumulation_steps
                    loss.backward()

                    effective_batch_loss += loss.item()
                    effective_batch_recon_loss += loss_recon.item()

                except StopIteration:
                    break

            optimizer.step()

            if warmup_scheduler is not None and not warmup_scheduler.is_warmup_complete():
                warmup_scheduler.step()

            avg_loss.append(effective_batch_loss)
            batch_losses.append(effective_batch_loss)
            avg_recon_loss.append(effective_batch_recon_loss)

            if effective_idx % 40 == 0 and writer is not None:
                global_step = epoch * num_effective_batches + effective_idx
                writer.add_scalar('Train_Loss_Batch', effective_batch_loss, global_step)
                writer.add_scalar('Train_Loss_Recon_Batch', effective_batch_recon_loss, global_step)

            pbar.set_postfix({'Loss': f'{np.mean(avg_loss):.3f}', 'Recon': f'{np.mean(avg_recon_loss):.3f}'})

        except Exception as e:
            msg = f"Error in training effective batch {effective_idx}: {e}"
            if logger:
                logger.error(msg)
            else:
                print(msg)
            continue

    epoch_loss = np.mean(avg_loss)
    epoch_metrics = metrics_accumulator.compute_epoch_stats()

    if logger:
        logger.info(f'Epoch {epoch}, Loss: {epoch_loss:.6f}')
    else:
        print(f'Epoch {epoch}, Loss: {epoch_loss:.6f}')
    print_metrics_summary(epoch_metrics, "  Train ", logger)

    return epoch_loss, epoch_metrics


def val(model, validloader, optimizer, device, config, writer, epoch, logger=None):
    """One validation pass."""
    if logger:
        logger.info('-------------validation--------------')
    else:
        print('-------------validation--------------')
    model.eval()

    metrics_accumulator = EpochMetricsAccumulator()

    with torch.no_grad():
        valid_error = []
        for idx, data in enumerate(validloader):
            try:
                myo_v, myo_f, myo_e, subid = data
                myo_v = myo_v.to(device)
                myo_f = myo_f.to(device)
                myo_e = myo_e.to(device)

                v_out, logvar, mu = model(myo_v, myo_f, myo_e)

                _, loss_recon = Loss.VAECELoss(v_out, myo_v, myo_f, logvar, mu,
                                               beta=config.beta, lambd=config.lambd,
                                               lambd_s=config.lambd_s, loss=config.loss)
                valid_error.append(loss_recon)

                try:
                    hd_values, assd_values = compute_batch_mesh_metrics(v_out, myo_v, subid, logger)
                    metrics_accumulator.update_raw_values(hd_values, assd_values)
                except Exception as e:
                    msg = f"Warning: metrics failed for validation batch {idx}: {e}"
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)

            except Exception as e:
                msg = f"Error in validation batch {idx}: {e}"
                if logger:
                    logger.error(msg)
                else:
                    print(msg)
                continue

        if valid_error:
            this_val_error = torch.mean(torch.stack(valid_error))
        else:
            msg = "Warning: no valid validation batches processed"
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            this_val_error = float('inf')

        epoch_metrics = metrics_accumulator.compute_epoch_stats()

        if logger:
            logger.info(f'Epoch {epoch}, Validation Error: {this_val_error:.6f}')
        else:
            print(f'Epoch {epoch}, Validation Error: {this_val_error:.6f}')
        print_metrics_summary(epoch_metrics, "  Val ", logger)
        if logger:
            logger.info('-------------------------------------')
        else:
            print('-------------------------------------')

        if writer is not None:
            writer.add_scalar('Val_HD_Mean', epoch_metrics['hd_mean'], epoch)
            writer.add_scalar('Val_HD_Median', epoch_metrics['hd_median'], epoch)
            writer.add_scalar('Val_ASSD_Mean', epoch_metrics['assd_mean'], epoch)
            writer.add_scalar('Val_ASSD_Median', epoch_metrics['assd_median'], epoch)

        return this_val_error, epoch_metrics


def log_hparams(writer, config, train_type, tag, final_train_loss=None, final_val_loss=None,
                best_val_loss=None, logger=None):
    """Write hyperparameter summary to TensorBoard."""
    hparams = {
        'train_type': train_type,
        'tag': tag,
        'learning_rate': config.lr,
        'batch_size': config.batch,
        'z_dim': config.z_dim,
        'n_epochs': config.n_epochs,
        'beta': config.beta,
        'lambd': config.lambd,
        'lambd_s': config.lambd_s,
        'loss_type': config.loss,
        'n_samples': config.n_samples,
        'seq_len': config.seq_len,
        'ff_size': config.ff_size,
        'num_heads': config.num_heads,
        'num_layers': config.num_layers,
        'activation': config.activation,
        'weight_decay': config.wd,
        'surf_type': config.surf_type,
        'val_freq': config.val_freq,
    }
    metrics = {
        'hparam/final_train_loss': final_train_loss if final_train_loss is not None else 0.0,
        'hparam/final_val_loss': final_val_loss if final_val_loss is not None else 0.0,
        'hparam/best_val_loss': best_val_loss if best_val_loss is not None else 0.0,
    }
    writer.add_hparams(hparams, metrics)

    if logger:
        logger.info(f"HParams logged (train={final_train_loss}, val={final_val_loss}, best={best_val_loss})")


def main(config):
    """Train the mesh-sequence VAE."""
    tag = config.tag
    model_dir = config.model_dir
    device = config.device
    train_type = config.train_type

    n_epochs = config.n_epochs
    n_samples = config.n_samples
    lr = config.lr
    z_dim = config.z_dim
    channal = 3

    seq_len = config.seq_len
    ff_size = config.ff_size
    num_heads = config.num_heads
    activation = config.activation
    num_layers = config.num_layers

    start = timeit.default_timer()

    checkpoint_info = load_checkpoint_and_setup_paths(config, None, None, device)
    model_name = checkpoint_info['model_name']
    start_epoch = checkpoint_info['start_epoch']
    best_val_loss = checkpoint_info['best_val_loss']
    logdir = checkpoint_info['logdir']
    cp_path = checkpoint_info['cp_path']
    best_model_path = checkpoint_info['best_model_path']
    intermediate_model_path = checkpoint_info['intermediate_model_path']
    is_resume = checkpoint_info['is_resume']

    setup_training_directories(logdir, cp_path)

    logger = setup_logger(cp_path, config)

    logger.info(f"channal (input dim): {channal}")
    logger.info(f"z_dim (latent dim): {z_dim}")
    logger.info(f"n_samples (points): {n_samples}")
    logger.info(f"train_type: {train_type}")

    try:
        trainset = MeshVAEDataset(config, 'train')
        validset = MeshVAEDataset(config, 'val')
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        return

    trainloader = DataLoader(trainset, batch_size=config.batch, shuffle=True, num_workers=config.num_workers)
    validloader = DataLoader(validset, batch_size=config.batch, shuffle=False, num_workers=config.num_workers)

    logger.info(f"Training samples: {len(trainset)}")
    logger.info(f"Validation samples: {len(validset)}")
    logger.info(f"Batch size: {config.batch}")

    try:
        model = mesh_vae_mod.MeshVAE(
            dim_in=channal,
            z_dim=z_dim,
            points=n_samples,
            seq_len=seq_len,
            ff_size=ff_size,
            num_heads=num_heads,
            activation=activation,
            num_layers=num_layers,
        ).to(device)
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return

    if config.wd:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    start_lr = lr / 10.0
    warmup_scheduler = WarmupScheduler(
        optimizer=optimizer,
        warmup_batches=config.warmup_batches,
        start_lr=start_lr,
        target_lr=lr,
    )

    batches_per_epoch = len(trainset) // (config.batch * config.accumulation_steps)
    plateau_patience_epochs = max(1, config.plateau_patience_batches // max(1, batches_per_epoch))

    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=config.lr_reduction_factor,
        patience=plateau_patience_epochs,
        min_lr=config.min_lr,
        verbose=True,
    )

    logger.info(f"Warmup batches: {config.warmup_batches}")
    logger.info(f"Warmup LR: {start_lr:.2e} -> {lr:.2e}")
    logger.info(f"Plateau patience: {config.plateau_patience_batches} batches (~{plateau_patience_epochs} epochs)")
    logger.info(f"LR reduction factor: {config.lr_reduction_factor}")
    logger.info(f"Minimum LR: {config.min_lr:.2e}")
    logger.info(f"Batches per epoch: ~{batches_per_epoch}")

    if is_resume:
        checkpoint_info = load_checkpoint_and_setup_paths(config, model, optimizer, device)
        start_epoch = checkpoint_info['start_epoch']
        best_val_loss = checkpoint_info['best_val_loss']

        if start_epoch > n_epochs:
            original_n_epochs = n_epochs
            n_epochs = start_epoch + original_n_epochs
            logger.warning(f"Checkpoint epoch ({start_epoch - 1}) exceeds n_epochs ({original_n_epochs})")
            logger.info(f"Auto-adjusting to train from {start_epoch} to {n_epochs}")

    writer = SummaryWriter(logdir)
    model.to(device)

    if is_resume:
        logger.info(f"Resuming from epoch {start_epoch} to {n_epochs}")
    else:
        logger.info(f"Training from epoch {start_epoch} to {n_epochs}")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")

    final_train_loss = None
    final_val_loss = None
    val_freq = config.val_freq

    for epoch in range(start_epoch, n_epochs + 1):
        epoch_start_time = time.time()

        train_loss, train_metrics = train(model, trainloader, optimizer, device, config, writer, epoch,
                                          warmup_scheduler, logger)
        final_train_loss = train_loss

        if epoch % val_freq == 0:
            val_loss, val_metrics = val(model, validloader, optimizer, device, config, writer, epoch, logger)
            final_val_loss = val_loss

            if warmup_scheduler.is_warmup_complete():
                plateau_scheduler.step(val_loss)

            if val_loss < best_val_loss:
                logger.info(f"New best validation loss: {best_val_loss:.6f} -> {val_loss:.6f}")
                best_val_loss = val_loss
                save_model(model, optimizer, epoch, train_loss, val_loss,
                           best_model_path, model_type="best", logger=logger)
            else:
                logger.info(f"Validation loss: {val_loss:.6f} (best: {best_val_loss:.6f})")

            if epoch % 10 == 0 and epoch > 0:
                save_model(model, optimizer, epoch, train_loss, val_loss,
                           intermediate_model_path, model_type="intermediate", logger=logger)

            try:
                writer.add_scalar('Train_Loss', train_loss, epoch)
                writer.add_scalar('Val_Loss', val_loss, epoch)
                writer.add_scalar('Best_Val_Loss', best_val_loss, epoch)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            except Exception as e:
                logger.error(f"Error writing to tensorboard: {e}")

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        logger.info(f"Epoch {epoch} completed in {epoch_duration // 60:.0f}:{epoch_duration % 60:.2f}")

    logger.info("Training completed.")
    logger.info(f"Final train loss: {final_train_loss:.6f}")
    logger.info(f"Final validation loss: {final_val_loss:.6f}")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Best model saved at: {best_model_path}")

    log_hparams(writer, config, train_type, tag, final_train_loss, final_val_loss, best_val_loss, logger)

    writer.add_scalar('Summary/Best_Val_Loss', best_val_loss, n_epochs)
    writer.add_scalar('Summary/Final_Train_Loss', final_train_loss, n_epochs)
    writer.add_scalar('Summary/Final_Val_Loss', final_val_loss, n_epochs)
    writer.add_scalar('Summary/Total_Epochs', n_epochs, n_epochs)

    writer.close()


if __name__ == '__main__':
    main(load_config())
