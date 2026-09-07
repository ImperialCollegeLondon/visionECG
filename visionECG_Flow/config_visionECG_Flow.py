import os
import torch
from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class VisionECGFlowConfig:
    # MODE
    mode: str = 'sequence_level' # 'sequence_level' | 'frame_resolved'

    num_frames: int = 50
    frame_embed_dim: int = 1024

    single_frame_training: bool = True
    frame_selection_strategy: str = 'random'

    # MODEL HYPERPARAMETERS
    dim_hid: int = 1024
    con_emb: int = 64
    ecg_emb: int = 512
    num_blocks: int = 8
    drop_rate: float = 0.25
    t_emb: int = 1024

    # TRAINING
    batch_size: int = 256
    learning_rate: float = 5e-5
    weight_decay: float = 0
    num_epochs: int = 600

    # Early stopping
    patience: int = 600
    min_delta: float = 1e-6

    template_strategy: str = 'mean'      

    # DATA
    train_csv_path: str = ''             
    val_csv_path: str = ''              

    normalize_data: bool = True

    demo_cols: List[str] = field(default_factory=lambda: [
        'Age', 'Sex', 'Weight', 'Height',
        'DBP_at_MRI', 'SBP_at_MRI', 'BMI', 'BSA',
    ])
    demographic_dim: int = 0              
    motion_embed_dim: int = 512
    ecg_embed_dim: int = 1024

    device: str = 'cuda'
    gpu_id: int = 0
    seed: int = 42
    num_workers: int = 4

    # OUTPUT PATHS
    base_output_dir: str = './outputs'
    output_dir: str = ''                  
    log_dir: str = ''                     
    run_id: str = ''                      

    def __post_init__(self):
        self.demographic_dim = len(self.demo_cols)

    def initialize_paths_and_device(self):
        """Resolve derived paths and device."""
        if self.mode not in ('sequence_level', 'frame_resolved'):
            raise ValueError(f"mode must be 'sequence_level' or 'frame_resolved', got {self.mode!r}")
        if self.num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {self.num_frames}")
        if self.frame_embed_dim <= 0:
            raise ValueError(f"frame_embed_dim must be positive, got {self.frame_embed_dim}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.run_id:
            self.run_id = f"{self.mode}_{timestamp}"

        self.output_dir = os.path.join(self.base_output_dir, self.run_id)
        self.log_dir = os.path.join(self.output_dir, 'logs')

        if self.device == 'cuda' and torch.cuda.is_available():
            if self.gpu_id >= torch.cuda.device_count():
                print(f"Warning: GPU {self.gpu_id} not available. Using GPU 0.")
                self.gpu_id = 0
            self.device = f'cuda:{self.gpu_id}'
        elif self.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = 'cpu'

    @property
    def is_frame_resolved(self) -> bool:
        return self.mode == 'frame_resolved'
        # return self.mode == '50frames'

    def to_dict(self) -> dict:
        return {
            'mode': self.mode,
            'num_frames': self.num_frames,
            'frame_embed_dim': self.frame_embed_dim,
            'single_frame_training': self.single_frame_training,
            'frame_selection_strategy': self.frame_selection_strategy,

            'dim_hid': self.dim_hid,
            'con_emb': self.con_emb,
            'ecg_emb': self.ecg_emb,
            'num_blocks': self.num_blocks,
            'drop_rate': self.drop_rate,
            't_emb': self.t_emb,

            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs,

            'patience': self.patience,
            'min_delta': self.min_delta,

            'template_strategy': self.template_strategy,

            'train_csv_path': self.train_csv_path,
            'val_csv_path': self.val_csv_path,

            'demo_cols': list(self.demo_cols),
            'demographic_dim': self.demographic_dim,
            'motion_embed_dim': self.motion_embed_dim,
            'ecg_embed_dim': self.ecg_embed_dim,
            'normalize_data': self.normalize_data,

            'device': self.device,
            'seed': self.seed,
            'run_id': self.run_id,
        }
