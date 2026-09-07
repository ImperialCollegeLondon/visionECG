import argparse
import datetime
import os
import time
import traceback

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import loss_visionECG_Motion as Loss
from config_visionECG_Motion import MotionDecoderConfig, load_config
from dataset_visionECG_Motion import FrameDataModule
from metrics_visionECG_Motion import (
    EpochMetricsAccumulator,
    compute_mesh_metrics_for_sequence,
    print_metrics_summary,
)
from model_visionECG_Motion import build_model_from_config, load_pretrained_decoder
from utils_visionECG_Motion import count_parameters, set_seed, setup_dir, setup_logging


def save_model(
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    checkpoint_name,
    config=None,
    model_type="best",
    logger=None,
):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch_num": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "model_type": model_type,
        "save_time": datetime.datetime.now().isoformat(),
        "config": config.to_dict() if config else None,
    }
    torch.save(checkpoint, checkpoint_name)
    msg = (
        f"{model_type.capitalize()} model saved: {checkpoint_name} | "
        f"Epoch: {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {float(val_loss):.6f}"
    )
    logger.info(msg) if logger else print(msg)


def load_checkpoint_for_resume(
    model,
    checkpoint_path,
    optimizer=None,
    resume_mode="full",
    device="cuda",
    logger=None,
):
    if not os.path.exists(checkpoint_path):
        msg = f"Checkpoint not found: {checkpoint_path}"
        logger.error(msg) if logger else print(msg)
        return 1, float("inf"), False

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)

        if "decoder.finallayer.0.weight" in state_dict:
            checkpoint_latent_dim = state_dict["decoder.finallayer.0.weight"].shape[1]
            if checkpoint_latent_dim != model.latent_dim:
                raise ValueError(
                    "Architecture mismatch: checkpoint latent_dim="
                    f"{checkpoint_latent_dim}, model latent_dim={model.latent_dim}"
                )

        model.load_state_dict(checkpoint["state_dict"])

        if resume_mode == "full":
            if optimizer is not None and "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint.get("epoch_num", 0) + 1
            best_val_loss = checkpoint.get("val_loss", float("inf"))
        elif resume_mode == "weights_only":
            start_epoch = 1
            best_val_loss = float("inf")
        else:
            raise ValueError(f"Invalid resume_mode: {resume_mode}")

        logger.info("Loaded checkpoint: %s", checkpoint_path) if logger else None
        return start_epoch, best_val_loss, True
    except Exception as exc:
        msg = f"Error loading checkpoint: {exc}"
        if logger:
            logger.error(msg)
            logger.error(traceback.format_exc())
        else:
            print(msg)
            traceback.print_exc()
        return 1, float("inf"), False


def compute_batch_mesh_metrics(v_out, v_gt, patient_ids):
    hd_values = []
    hd90_values = []
    assd_values = []
    for batch_idx in range(v_out.shape[0]):
        try:
            avg_hd, avg_hd90, avg_assd = compute_mesh_metrics_for_sequence(
                v_out[batch_idx],
                v_gt[batch_idx],
            )
            hd_values.append(avg_hd.item())
            hd90_values.append(avg_hd90.item())
            assd_values.append(avg_assd.item())
        except Exception as exc:
            patient_id = patient_ids[batch_idx] if patient_ids is not None else batch_idx
            print(f"Warning: error computing metrics for patient {patient_id}: {exc}")
    return hd_values, hd90_values, assd_values


def train(
    model,
    trainloader,
    optimizer,
    device,
    config,
    writer,
    epoch,
    global_step,
    logger=None,
):
    model.train()
    avg_loss = []
    batch_losses = []
    metrics_accumulator = EpochMetricsAccumulator()
    accumulation_steps = config.accumulation_steps
    optimizer.zero_grad()

    pbar = tqdm(trainloader, desc=f"Epoch {epoch} Training")
    for idx, data in enumerate(pbar):
        try:
            z = data["z"].to(device)
            queries = data["queries"].to(device)
            heart_v = data["heart_v"].to(device)
            heart_f = data["heart_f"].to(device)
            demographics = data["demographics"].to(device)
            subid = data["eid"]

            v_out = model(z, queries, demographics)
            loss, loss_recon = Loss.VAECELoss(
                v_out,
                heart_v,
                heart_f,
                logvar=None,
                mu=None,
                beta=0.0,
                lambd=config.lambd,
                lambd_s=config.lambd_s,
                loss=config.loss,
            )

            loss = loss / accumulation_steps
            avg_loss.append(loss.item() * accumulation_steps)
            batch_losses.append(loss.item() * accumulation_steps)

            hd_values, hd90_values, assd_values = [], [], []
            compute_metrics = False
            if config.compute_train_metrics_freq and config.compute_train_metrics_freq > 0:
                compute_metrics = (
                    config.compute_train_metrics_freq == -1
                    or idx % config.compute_train_metrics_freq == 0
                )
            if compute_metrics:
                with torch.no_grad():
                    hd_values, hd90_values, assd_values = compute_batch_mesh_metrics(
                        v_out,
                        heart_v,
                        subid,
                    )
                    metrics_accumulator.update_raw_values(
                        hd_values,
                        hd90_values,
                        assd_values,
                    )

            loss.backward()
            if (idx + 1) % accumulation_steps == 0:
                if config.grad_clip_value:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.grad_clip_value,
                    )
                optimizer.step()
                optimizer.zero_grad()

            if idx % 10 == 0 and writer is not None:
                batch_step = global_step * len(trainloader) + idx
                writer.add_scalar("Batch/Train_Loss", loss.item(), batch_step)
                writer.add_scalar("Batch/Train_Loss_Recon", loss_recon.item(), batch_step)
                if hd_values and hd90_values and assd_values:
                    writer.add_scalar("Batch/Train_HD", np.mean(hd_values), batch_step)
                    writer.add_scalar(
                        "Batch/Train_HD90",
                        np.mean(hd90_values),
                        batch_step,
                    )
                    writer.add_scalar("Batch/Train_ASSD", np.mean(assd_values), batch_step)

            pbar.set_postfix(
                {
                    "Loss": f"{loss.item() * accumulation_steps:.3f}",
                    "Loss_recon": f"{loss_recon.item():.3f}",
                    "Batch Avg Loss": f"{np.mean(batch_losses):.3f}",
                }
            )
        except Exception as exc:
            msg = f"Error in training batch {idx}: {exc}"
            if logger:
                logger.error(msg)
                logger.error(traceback.format_exc())
            else:
                print(msg)
                traceback.print_exc()
            continue

    if len(trainloader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    epoch_loss = float(np.mean(avg_loss)) if avg_loss else float("inf")
    epoch_metrics = metrics_accumulator.compute_epoch_stats()
    logger.info("Epoch %s, Loss: %.6f", epoch, epoch_loss) if logger else print(epoch_loss)
    if epoch_metrics["num_samples"] > 0:
        print_metrics_summary(epoch_metrics, "  Train ", logger=logger)
    else:
        logger.info("  Train metrics disabled") if logger else print("Train metrics disabled")

    if writer is not None:
        writer.add_scalar("Train_Loss_Std", np.std(batch_losses), epoch)
        if epoch_metrics["num_samples"] > 0:
            writer.add_scalar("Train_HD_Mean", epoch_metrics["hd_mean"], epoch)
            writer.add_scalar("Train_HD90_Mean", epoch_metrics["hd90_mean"], epoch)
            writer.add_scalar("Train_ASSD_Mean", epoch_metrics["assd_mean"], epoch)

    return epoch_loss, epoch_metrics


def val(model, validloader, device, config, writer, global_step, logger=None):
    logger.info("-------------validation--------------") if logger else print("validation")
    model.eval()
    metrics_accumulator = EpochMetricsAccumulator()
    valid_error = []

    with torch.no_grad():
        for idx, data in enumerate(validloader):
            try:
                z = data["z"].to(device)
                queries = data["queries"].to(device)
                heart_v = data["heart_v"].to(device)
                heart_f = data["heart_f"].to(device)
                demographics = data["demographics"].to(device)
                subid = data["eid"]

                v_out = model(z, queries, demographics)
                loss, loss_recon = Loss.VAECELoss(
                    v_out,
                    heart_v,
                    heart_f,
                    logvar=None,
                    mu=None,
                    beta=0.0,
                    lambd=config.lambd,
                    lambd_s=config.lambd_s,
                    loss=config.loss,
                )
                valid_error.append(loss_recon.detach())

                hd_values, hd90_values, assd_values = compute_batch_mesh_metrics(
                    v_out,
                    heart_v,
                    subid,
                )
                metrics_accumulator.update_raw_values(
                    hd_values,
                    hd90_values,
                    assd_values,
                )
            except Exception as exc:
                msg = f"Error in validation batch {idx}: {exc}"
                if logger:
                    logger.error(msg)
                    logger.error(traceback.format_exc())
                else:
                    print(msg)
                continue

    if valid_error:
        this_val_error = torch.mean(torch.stack(valid_error))
    else:
        this_val_error = torch.tensor(float("inf"), device=device)

    epoch_metrics = metrics_accumulator.compute_epoch_stats()
    logger.info(
        "Step %s, Validation Error: %.6f",
        global_step,
        this_val_error.item(),
    ) if logger else print(this_val_error.item())
    print_metrics_summary(epoch_metrics, "  Val ", logger=logger)

    if writer is not None:
        writer.add_scalar("Val/HD_Mean", epoch_metrics["hd_mean"], global_step)
        writer.add_scalar("Val/HD90_Mean", epoch_metrics["hd90_mean"], global_step)
        writer.add_scalar(
            "Val/ASSD_Mean",
            epoch_metrics["assd_mean"],
            global_step,
        )

    return this_val_error, epoch_metrics


def select_metric(metrics, val_loss, config):
    if config.validation_metric == "hd":
        return metrics["hd_mean"]
    if config.validation_metric == "hd90":
        return metrics["hd90_mean"]
    if config.validation_metric == "assd":
        return metrics["assd_mean"]
    raise ValueError(f"Unsupported validation_metric: {config.validation_metric}")


def maybe_save_best(
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    val_metrics,
    best_val_loss,
    best_val_metric,
    best_model_path,
    config,
    logger,
):
    current_metric = select_metric(val_metrics, val_loss, config)
    current_loss = float(val_loss.item() if torch.is_tensor(val_loss) else val_loss)

    improved = current_metric < best_val_metric
    if improved:
        best_val_metric = current_metric
        best_val_loss = current_loss
        save_model(
            model,
            optimizer,
            epoch,
            train_loss,
            current_loss,
            best_model_path,
            config=config,
            model_type="best",
            logger=logger,
        )
        logger.info(
            "New best %s: %.6f",
            config.validation_metric,
            best_val_metric,
        ) if logger else None
    return best_val_loss, best_val_metric


def main(config: MotionDecoderConfig):
    set_seed(config.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    device = config.device

    cp_path = os.path.join(config.output_dir, "checkpoints")
    best_model_path = os.path.join(cp_path, "best_model.pt")

    logger = setup_logging(config.log_dir)
    setup_dir(cp_path)
    if config.use_tensorboard:
        setup_dir(config.tensorboard_log_dir)

    logger.info("=" * 80)
    logger.info("visionECG_Motion Training")
    logger.info("=" * 80)
    logger.info("Input: z + per-frame queries + demographics")
    logger.info("Conditioning: demographics -> z residual only")
    logger.info("Frame residuals: query latents only")
    for k, v in config.to_dict().items():
        if (k.endswith("_path") or k.endswith("_dir")
                or k.endswith("_checkpoint")
                or k.startswith(("memory_z_", "queries_", "target_seg"))):
            v = "<hidden>" if v else "<not set>"
        logger.info("  %s: %s", k, v)

    data_module = FrameDataModule(config)
    data_module.setup("fit")
    trainloader = data_module.train_dataloader()
    validloader = data_module.val_dataloader()

    model = build_model_from_config(config).to(device)
    if config.load_pretrained and not config.resume_checkpoint:
        model = load_pretrained_decoder(
            model,
            config.mesh_decoder_checkpoint,
            device,
        )
        model = model.to(device)

    param_counts = count_parameters(model)
    logger.info("Model parameters: %s", param_counts)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_metric = float("inf")
    if config.resume_checkpoint:
        start_epoch, best_val_loss, restored = load_checkpoint_for_resume(
            model=model,
            checkpoint_path=config.resume_checkpoint,
            optimizer=optimizer,
            resume_mode=config.resume_mode,
            device=device,
            logger=logger,
        )
        if not restored:
            raise RuntimeError("Failed to load resume checkpoint")

    writer = SummaryWriter(config.tensorboard_log_dir) if config.use_tensorboard else None

    final_train_loss = float("nan")
    final_val_loss = float("nan")
    for epoch in range(start_epoch, config.n_epochs + 1):
        epoch_start_time = time.time()

        train_loss, _ = train(
            model,
            trainloader,
            optimizer,
            device,
            config,
            writer,
            epoch,
            epoch,
            logger,
        )
        val_loss, val_metrics = val(
            model,
            validloader,
            device,
            config,
            writer,
            epoch,
            logger,
        )
        best_val_loss, best_val_metric = maybe_save_best(
            model,
            optimizer,
            epoch,
            train_loss,
            val_loss,
            val_metrics,
            best_val_loss,
            best_val_metric,
            best_model_path,
            config,
            logger,
        )

        final_train_loss = train_loss
        final_val_loss = float(val_loss.item() if torch.is_tensor(val_loss) else val_loss)

        if epoch % config.save_every_n_epochs == 0:
            periodic_checkpoint_path = os.path.join(cp_path, f"checkpoint_epoch_{epoch}.pt")
            save_model(
                model,
                optimizer,
                epoch,
                train_loss,
                final_val_loss,
                periodic_checkpoint_path,
                config=config,
                model_type=f"epoch_{epoch}",
                logger=logger,
            )

        epoch_duration = time.time() - epoch_start_time
        logger.info(
            "Epoch %s completed in %.0f:%.2f",
            epoch,
            epoch_duration // 60,
            epoch_duration % 60,
        )

    logger.info("=" * 80)
    logger.info("visionECG_Motion training completed")
    logger.info("Final train loss: %.6f", final_train_loss)
    logger.info("Final val loss: %.6f", final_val_loss)
    logger.info("Best %s: %.6f", config.validation_metric, best_val_metric)
    logger.info("Best model: %s", best_model_path)
    logger.info("=" * 80)

    if writer is not None:
        writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train visionECG_Motion")
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--run_id", type=str)
    parser.add_argument("--resume_checkpoint", type=str)
    parser.add_argument("--resume_mode", type=str, choices=["full", "weights_only"])
    parser.add_argument("--accumulation_steps", type=int)
    parser.add_argument("--validation_metric", type=str, choices=["hd", "hd90", "assd"])
    parser.add_argument("--demo_cols", type=str, nargs="+")
    parser.add_argument("--no_tensorboard", dest="use_tensorboard", action="store_false")
    parser.add_argument("--no_pretrained", dest="load_pretrained", action="store_false")
    # Data paths
    parser.add_argument("--memory_z_train", type=str)
    parser.add_argument("--memory_z_val", type=str)
    parser.add_argument("--memory_z_test", type=str)

    parser.add_argument("--queries_train", type=str)
    parser.add_argument("--queries_val", type=str)
    parser.add_argument("--queries_test", type=str)

    parser.add_argument("--train_csv_path", type=str)
    parser.add_argument("--val_csv_path", type=str)
    parser.add_argument("--test_csv_path", type=str)

    parser.add_argument("--target_seg_dir", type=str)

    # Pretrained checkpoint
    parser.add_argument("--mesh_decoder_checkpoint", type=str)

    # Output
    parser.add_argument("--base_output_dir", type=str)

    parser.set_defaults(
        use_tensorboard=None,
        load_pretrained=None,
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(demo_cols=args.demo_cols)
    for key, value in vars(args).items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    if args.run_id is not None:
        cfg.run_id = args.run_id
    if args.use_tensorboard is not None:
        cfg.use_tensorboard = args.use_tensorboard
    if args.load_pretrained is not None:
        cfg.load_pretrained = args.load_pretrained
    cfg.__post_init__()
    main(cfg)
