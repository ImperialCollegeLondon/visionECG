import os
import datetime
import torch


def setup_dir(dir_path):
    """Create a directory if it does not exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def extract_model_name_from_checkpoint(checkpoint_path):
    """Return the model directory name given a checkpoint path."""
    model_dir = os.path.dirname(checkpoint_path)
    return os.path.basename(model_dir)


def load_checkpoint_and_setup_paths(config, model, optimizer, device):
    """Resolve paths and optionally restore a checkpoint."""
    train_type = config.train_type
    z_dim = config.z_dim
    model_dir = config.model_dir
    checkpoint_file = config.checkpoint_file or None

    start_epoch = 0
    best_val_loss = float('inf')
    is_resume = False

    if checkpoint_file and os.path.exists(checkpoint_file):
        model_name = extract_model_name_from_checkpoint(checkpoint_file)
        is_resume = True
        print(f"Resuming training from checkpoint: {checkpoint_file}")
        print(f"   Model name: {model_name}")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_model_name = (
            f"{train_type}_z_dim{z_dim}_loss_{config.loss}_beta{config.beta}_"
            f"lambd{config.lambd}_lambds{config.lambd_s}_lr{config.lr}_wd{config.wd}_batch{config.batch}"
        )
        model_name = f"{base_model_name}_{timestamp}"
        print(f"Starting new training. Model name: {model_name}")

    logdir = f"{model_dir}/tb/{model_name}"
    cp_path = f"{model_dir}/model/{model_name}"
    best_model_path = f"{cp_path}/best_model.pt"
    intermediate_model_path = f"{cp_path}/intermediate_checkpoint.pt"

    if checkpoint_file:
        try:
            if os.path.exists(checkpoint_file):
                checkpoint_path = checkpoint_file
            elif os.path.exists(best_model_path):
                checkpoint_path = best_model_path
            elif os.path.exists(intermediate_model_path):
                checkpoint_path = intermediate_model_path
            else:
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            start_epoch = checkpoint['epoch_num'] + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))

            if model is not None and optimizer is not None:
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                print(f"Loaded checkpoint from epoch {checkpoint['epoch_num']}, best val loss {best_val_loss:.6f}")
                print(f"Resuming from epoch {start_epoch}")
            else:
                print(f"Checkpoint metadata loaded (epoch {checkpoint['epoch_num']}, best val {best_val_loss:.6f})")

        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            start_epoch = 0
            best_val_loss = float('inf')
            is_resume = False

    return {
        'model_name': model_name,
        'start_epoch': start_epoch,
        'best_val_loss': best_val_loss,
        'logdir': logdir,
        'cp_path': cp_path,
        'best_model_path': best_model_path,
        'intermediate_model_path': intermediate_model_path,
        'is_resume': is_resume,
    }


def setup_training_directories(logdir, cp_path):
    """Create the TensorBoard and checkpoint directories."""
    setup_dir(logdir)
    setup_dir(cp_path)
    print(f"TensorBoard logs: {logdir}")
    print(f"Model checkpoints: {cp_path}")
