import pathlib
import argparse
import multiprocessing
from datetime import timedelta
import typing

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import optuna

from dataset import (
    SpeakerDataset,
    open_hdf5_file,
)
from model.SpeechToExpressionModel import SpeechToExpressionModel
import config

        
tpu_available = False
try:
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.core.xla_model as xm
    if xm.xla_device_hw() == "TPU":
        preferred_device = "tpu"
        num_devices = 8  # Use 8 devices for TPU
        tpu_available = True
    else:
        preferred_device = "gpu" if torch.cuda.is_available() else "cpu"
        num_devices = 1  # Default to 1 device for GPU or CPU
except ImportError:
    # Fallback to GPU or CPU if TPU is not available
    preferred_device = "gpu" if torch.cuda.is_available() else "cpu"
    num_devices = 1  # Default to 1 device


if typing.TYPE_CHECKING:
    from dataset import (
        Batch,  # noqa: F401
        BatchData,  # noqa: F401
    )
##################################################################################################

WAV2VEC2_EMBED_DIM = 768
FACIAL_FEATURE_DIM = 31


##################################################################################################
def parse_args():
    # type: () -> argparse.Namespace
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Speech to Expression model.")
    
    # Dataset and model arguments
    parser.add_argument("--hdf5_file", type=str, required=True, help="Path to the HDF5 file.")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to the checkpoint to resume training.")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training.")

    # Model hyperparameters
    parser.add_argument("--config_file", type=str, default=None, help="Path to the configuration file.")

    parser.add_argument("--tune", action="store_true", help="Tune hyperparameters using Optuna.", default=False)
    
    return parser.parse_args()


##################################################################################################
class StopIfNotUnderThresholdByEpoch(pl.Callback):
    """PyTorch Lightning callback to stop training if the validation loss is not under a threshold by a certain epoch."""

    def __init__(self, target_epoch: int, threshold: float, monitor: str = "val_loss"):
        super().__init__()
        self.target_epoch = target_epoch
        self.threshold = threshold
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch == self.target_epoch - 1:
            current_val_loss = trainer.callback_metrics.get(self.monitor, None)
            if current_val_loss is not None:
                if current_val_loss >= self.threshold:
                    trainer.should_stop = True


def objective(trial):
    # type: (optuna.Trial) -> float
    """Objective function for Optuna to optimize hyperparameters."""

    # time_embed_dim = trial.suggest_int("time_embed_dim", 32, 2048)
    time_embed_dim = 512

    # Suggest hyperparameters to tune
    hparams = config.SpeechToExpressionConfig(
        model="speech_to_expression",
        embed_dim=WAV2VEC2_EMBED_DIM,
        time_embed_dim=time_embed_dim,
        output_dim=FACIAL_FEATURE_DIM,
        # lr = 5e-05,
        lr = trial.suggest_float("lr", 1e-7, 1e-3),
        st=config.ShortTermConfig(
            prev_window=3,
            next_window=6,
            # head_num=8,
            head_num=trial.suggest_int("st_head_num", 1, 16),
        ),
        lt=config.LongTermConfig(
            prev_window=50,
            next_window=150,
            head_num=trial.suggest_int("lt_head_num", 1, 16),
            layer_num=trial.suggest_int("lt_layer_num", 1, 8),
            # head_num=8,
            # layer_num=2,
        ),
        diffusion=config.DiffusionConfig(
            loss_type="l1",
            noise_decoder_config=config.NoiseDecoderConfig(
                head_num=trial.suggest_int("noise_head_num", 1, 32),
                hidden_dim=trial.suggest_int("noise_hidden_dim", 32, 2048),
                layer_num=trial.suggest_int("noise_layer_num", 1, 8),
                dim_feedforward=trial.suggest_int("noise_dim_feedforward", 32, 4096),
                dropout=trial.suggest_float("noise_dropout", 0.1, 0.5),
                # head_num=10,
                # hidden_dim=1024,
                # layer_num=1,
                # dim_feedforward=4096,
                # dropout=0.1,
            ),
            train_timesteps_num=1000,
        ),
    )
    
    # Pass the suggested hyperparameters into your model
    model = SpeechToExpressionModel(hparams)
    val_loss_checkpoint = ModelCheckpoint(
        monitor="val_loss", 
        filename="best_model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min"
    )

    time_checkpoint = ModelCheckpoint(
        filename="best_model-{epoch:02d}",
        train_time_interval=timedelta(minutes=30),
    )

    stop_if_not_under_threshold = StopIfNotUnderThresholdByEpoch(2, 0.8)

    # Create PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=8,
        logger=TensorBoardLogger("../logs/", name="optuna20241208_02"),
        callbacks=[
            val_loss_checkpoint,
            time_checkpoint,
            stop_if_not_under_threshold,
            EarlyStopping(monitor="val_loss")
        ],
        accumulate_grad_batches=3,
        # gradient_clip_val=0.5,

        accelerator=preferred_device,
        devices=num_devices,
    )

    # batch_size = trial.suggest_int("batch_size", 1, 8)
    batch_size = 4

    # Create DataLoaders with the suggested batch size
    args = parse_args()
    train_loader, val_loader = prepare_dataloaders(
        args.hdf5_file,
        hparams.st.prev_window,
        hparams.st.next_window,
        hparams.lt.prev_window,
        hparams.lt.next_window,
        batch_size,
        
    )
    trainer.fit(model, train_loader, val_loader)

    # Return the validation loss (or accuracy, depending on your goal)
    return trainer.callback_metrics["val_loss"].item()


def show_best_parameters(study):
    # type: (optuna.Study) -> None
    """Show the best hyperparameters found by Optuna."""
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


##################################################################################################


def prepare_dataloaders(
        hdf5_file,
        prev_short_term_window,
        next_short_term_window,
        prev_long_term_window,
        next_long_term_window,
        batch_size,
):
    # type: (str, int, int, int, int, int) -> tuple[DataLoader, DataLoader]
    """Prepares DataLoader for training and validation."""

    # Initialize dataset and dataloader
    dataset = SpeakerDataset(
        hdf5_file=hdf5_file, 
        embed_dim=WAV2VEC2_EMBED_DIM,
        prev_short_term_window=prev_short_term_window, 
        next_short_term_window=next_short_term_window, 
        prev_long_term_window=prev_long_term_window,
        next_long_term_window=next_long_term_window,
    )

    num_workers = multiprocessing.cpu_count() // 2
    
    # Train-test split (simple version, adjust as needed)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=open_hdf5_file,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=open_hdf5_file,
        persistent_workers=True
    )

    if tpu_available:
        train_loader = pl.MpDeviceLoader(train_loader, xm.xla_device())  # type: ignore
        val_loader = pl.MpDeviceLoader(val_loader, xm.xla_device())  # type: ignore
    
    return train_loader, val_loader


def main():
    # Parse arguments
    args = parse_args()

    if args.tune:
        # Optimize hyperparameters using Optuna
        study = optuna.create_study(
            direction="minimize",
            storage="sqlite:///optuna3.db",
            study_name="speech_to_expression_2024-12-08_02",
            load_if_exists=True
        )
        study.optimize(objective, n_trials=100)
        show_best_parameters(study)
        return

    if args.config_file:
        # Load hyperparameters from a configuration file
        hparams = config.load_from_yaml(args.config_file)

    else:
        hparams = config.SpeechToExpressionConfig(
            model="speech_to_expression",
            embed_dim=WAV2VEC2_EMBED_DIM,
            time_embed_dim=100,
            output_dim=FACIAL_FEATURE_DIM,
            lr=args.learning_rate,
            st=config.ShortTermConfig(
                prev_window=3,
                next_window=6,
                head_num=2,
            ),
            lt=config.LongTermConfig(
                prev_window=90,
                next_window=60,
                head_num=2,
                layer_num=3,
            ),
            diffusion=config.DiffusionConfig(
                model="ddpm",
                loss_type="l1",
                noise_decoder_config=config.NoiseDecoderConfig(
                    head_num=16,
                    hidden_dim=1024,
                    layer_num=5,
                ),
                train_timesteps_num=1000,
            ),
        )

    # Prepare the dataset and dataloaders
    train_loader, val_loader = prepare_dataloaders(
        args.hdf5_file,
        hparams.st.prev_window,
        hparams.st.next_window,
        hparams.lt.prev_window,
        hparams.lt.next_window,
        args.batch_size,
    )

    if args.resume_checkpoint:
        # Load the model from a checkpoint
        model = SpeechToExpressionModel.load_from_checkpoint(args.resume_checkpoint)

    else:
        # Initialize the model
        model = SpeechToExpressionModel(hparams)

    print("training...")
    print("batch_size: ", args.batch_size)
    print("learning_rate: ", args.learning_rate)
    print("num_epochs: ", args.num_epochs)
    print("config_file: ", args.config_file)
    print("hparams: ", hparams)

    # Set up logging using TensorBoard
    log_dir = pathlib.Path(__file__).parent.parent.parent / "logs"
    logger = TensorBoardLogger(log_dir.as_posix(), name="diffusion")

    # Set up checkpoints to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", 
        filename="best_model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=4,
        mode="min"
    )

    # Set up early stopping if needed (optional)
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator=preferred_device,
        devices=num_devices,
        logger=logger,
        # TPUs may require you to adjust your batch size for efficient usage.
        # If you're running into memory issues, you can use gradient accumulation
        # to simulate larger batch sizes: Add the accumulate_grad_batches option
        # to your trainer to control how many batches are accumulated before performing a backward pass:
        accumulate_grad_batches=4,  # Accumulate gradients over 4 batches
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
