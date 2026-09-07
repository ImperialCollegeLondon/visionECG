import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import torch


@dataclass
class MotionDecoderConfig:
    """Configuration for bottleneck-conditioned frame decoder training."""

    key_notes: str = (
        "visionECG_Motion"
    )

    # Model architecture
    latent_dim: int = 512
    seq_len: int = 50
    points: int = 1412
    ff_size: int = 2048
    num_layers: int = 4
    num_heads: int = 8
    activation: str = "gelu"
    decoder_dropout: float = 0.1

    # Residual dimensions and bottleneck demographics conditioning
    residual_hidden_dim: int = 1024
    residual_dropout: float = 0.1
    demo_cols: Optional[List[str]] = None
    demographic_dim: int = 0
    con_emb: int = 64

    # Pretrained MeshVAE checkpoint
    mesh_decoder_checkpoint: str = ""
    load_pretrained: bool = True

    # Data paths
    memory_z_train: str = ""
    memory_z_val: str = ""
    memory_z_test: str = ""
    queries_train: str = ""
    queries_val: str = ""
    queries_test: str = ""

    train_csv_path: str = ""
    val_csv_path: str = ""
    test_csv_path: str = ""

    target_seg_dir: str = ""
    surf_type: str = "all"

    # Training hyperparameters
    batch_size: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    n_epochs: int = 100

    resume_checkpoint: str = None
    resume_mode: str = "weights_only"

    accumulation_steps: int = 32

    grad_clip_value: float = 1.0

    # Loss configuration.
    lambd: float = 1.0
    lambd_s: float = 1.0
    loss: str = "cham_smooth"

    # Validation and logging
    save_every_n_epochs: int = 10
    validation_metric: str = "hd"
    compute_train_metrics_freq: int = 0

    # System
    device: str = "cuda"
    gpu_id: int = 0
    seed: int = 42
    num_workers: int = 1
    use_tensorboard: bool = True

    # Output paths
    base_output_dir: str = "./outputs"
    output_dir: str = ""
    model_save_path: str = ""
    log_dir: str = ""
    tensorboard_log_dir: str = ""
    run_id: str = ""

    def _get_abbreviated_params(self) -> str:
        h = f"h{self.latent_dim}"
        s = f"s{self.seq_len}"
        l = f"l{self.num_layers}"
        n = f"n{self.num_heads}"
        r = f"r{self.residual_hidden_dim}"
        c = f"c{self.con_emb}"
        bs = f"bs{self.batch_size}"
        lr_exp = (
            f"{self.learning_rate:.0e}"
            .replace("e-0", "e")
            .replace("e+0", "e")
            .replace("-", "")
        )
        lr = f"lr{lr_exp}"
        return f"{h}_{s}_{l}_{n}_{r}_{c}_{bs}_{lr}_{self.loss}"

    def __post_init__(self):
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.demo_cols is None or len(self.demo_cols) == 0:
            raise ValueError(
                "demo_cols must be provided (e.g., --demo_cols Age Sex Weight ...)"
            )
        self.demographic_dim = len(self.demo_cols)
        if self.con_emb <= 0:
            raise ValueError(f"con_emb must be positive, got {self.con_emb}")

        expected_residual_dim = self.latent_dim * 2
        if self.residual_hidden_dim != expected_residual_dim:
            print(
                "Warning: residual_hidden_dim "
                f"({self.residual_hidden_dim}) != 2 * latent_dim "
                f"({expected_residual_dim}); auto-adjusting."
            )
            self.residual_hidden_dim = expected_residual_dim

        if not self.run_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"motion_decoder_{self._get_abbreviated_params()}_{timestamp}"

        self.output_dir = os.path.join(self.base_output_dir, self.run_id)
        self.log_dir = os.path.join(self.output_dir, "logs")
        self.model_save_path = os.path.join(self.output_dir, "best_model.pt")
        if self.use_tensorboard:
            self.tensorboard_log_dir = os.path.join(
                self.base_output_dir, "tensorboard_logs", self.run_id
            )

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        if self.use_tensorboard:
            os.makedirs(self.tensorboard_log_dir, exist_ok=True)

        if self.device == "cuda" and torch.cuda.is_available():
            if self.gpu_id >= torch.cuda.device_count():
                print(f"Warning: GPU {self.gpu_id} not available. Using GPU 0.")
                self.gpu_id = 0
            self.device = f"cuda:{self.gpu_id}"
        elif self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"

    def to_dict(self) -> dict:
        return {
            "key_notes": self.key_notes,
            "latent_dim": self.latent_dim,
            "seq_len": self.seq_len,
            "points": self.points,
            "ff_size": self.ff_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "activation": self.activation,
            "decoder_dropout": self.decoder_dropout,
            "residual_hidden_dim": self.residual_hidden_dim,
            "residual_dropout": self.residual_dropout,
            "demo_cols": list(self.demo_cols),
            "demographic_dim": self.demographic_dim,
            "con_emb": self.con_emb,
            "mesh_decoder_checkpoint": self.mesh_decoder_checkpoint,
            "load_pretrained": self.load_pretrained,
            "memory_z_train": self.memory_z_train,
            "memory_z_val": self.memory_z_val,
            "memory_z_test": self.memory_z_test,
            "queries_train": self.queries_train,
            "queries_val": self.queries_val,
            "queries_test": self.queries_test,
            "train_csv_path": self.train_csv_path,
            "val_csv_path": self.val_csv_path,
            "test_csv_path": self.test_csv_path,
            "target_seg_dir": self.target_seg_dir,
            "surf_type": self.surf_type,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "n_epochs": self.n_epochs,
            "resume_checkpoint": self.resume_checkpoint,
            "resume_mode": self.resume_mode,
            "accumulation_steps": self.accumulation_steps,
            "grad_clip_value": self.grad_clip_value,
            "lambd": self.lambd,
            "lambd_s": self.lambd_s,
            "loss": self.loss,
            "save_every_n_epochs": self.save_every_n_epochs,
            "validation_metric": self.validation_metric,
            "compute_train_metrics_freq": self.compute_train_metrics_freq,
            "device": self.device,
            "gpu_id": self.gpu_id,
            "seed": self.seed,
            "num_workers": self.num_workers,
            "use_tensorboard": self.use_tensorboard,
            "base_output_dir": self.base_output_dir,
            "output_dir": self.output_dir,
            "model_save_path": self.model_save_path,
            "log_dir": self.log_dir,
            "tensorboard_log_dir": self.tensorboard_log_dir,
            "run_id": self.run_id,
        }


def load_config(demo_cols: Optional[List[str]] = None) -> MotionDecoderConfig:
    return MotionDecoderConfig(demo_cols=demo_cols)
