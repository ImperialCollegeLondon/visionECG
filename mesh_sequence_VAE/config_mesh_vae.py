"""Argparse configuration for the mesh-sequence VAE."""

import argparse
import torch


def load_config():
    parser = argparse.ArgumentParser(description='MeshVAE for cardiac motion sequences')

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--val_freq', type=int, default=1)

    parser.add_argument('--warmup_batches', type=int, default=0)
    parser.add_argument('--plateau_patience_batches', type=int, default=4000)
    parser.add_argument('--lr_reduction_factor', type=float, default=0.1)
    parser.add_argument('--min_lr', type=float, default=1e-4)
    parser.add_argument('--accumulation_steps', type=int, default=4)

    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--n_samples', type=int, default=1412)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--ff_size', type=int, default=2048)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--activation', type=str, default='gelu')

    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--lambd', type=float, default=1.0)
    parser.add_argument('--lambd_s', type=float, default=1.0)
    parser.add_argument('--loss', type=str, default='cham_smooth')

    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--label_dir', type=str, default='')
    parser.add_argument('--target_seg_dir', type=str, default='')

    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--surf_type', type=str, default='all', choices=['all', 'sample'])

    parser.add_argument('--train_type', type=str, default='mesh_vae')
    parser.add_argument('--tag', type=str, default='mesh_vae')

    parser.add_argument('--checkpoint_file', type=str, default='')

    args = parser.parse_args()

    missing = [name for name, val in
               (('model_dir', args.model_dir),
                ('label_dir', args.label_dir),
                ('target_seg_dir', args.target_seg_dir))
               if not val]
    if missing:
        parser.error(f"Required argument(s) missing: {', '.join('--' + m for m in missing)}")

    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    return args
