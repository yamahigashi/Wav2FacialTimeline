import pathlib
import argparse
import multiprocessing

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import (
    SpeakerDataset,
    open_hdf5_file,
)
from model import SpeechToExpressionModel

import typing
if typing.TYPE_CHECKING:
    from dataset import (
        Batch,  # noqa: F401
        BatchData,  # noqa: F401
    )


WAV2VEC2_EMBED_DIM = 768
FACIAL_FEATURE_DIM = 31


def parse_args():
    # type: () -> argparse.Namespace
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Speech to Expression model.")
    
    # Dataset and model arguments
    parser.add_argument("--hdf5_file", type=str, required=True, help="Path to the HDF5 file.")
    parser.add_argument("--prev_short_term_window", type=int, default=4, help="Number of short-term frames.")
    parser.add_argument("--next_short_term_window", type=int, default=3, help="Number of short-term frames.")
    parser.add_argument("--prev_long_term_window", type=int, default=90, help="Number of long-term frames.")
    parser.add_argument("--next_long_term_window", type=int, default=60, help="Number of long-term frames.")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to the checkpoint to resume training.")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of diffusion steps.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of Transformer layers.")
    
    return parser.parse_args()


def collate_fn(batch):
    # type: (list[Batch]) -> BatchData
    """Collate function to pad and stack the batch of sequences.
    Padding is done to ensure that all sequences have the same length. And
    masks are created to ignore the padded elements during training.
    
    Args:
        batch (list): List of tuples containing short-term features, long-term features, and labels.

    Returns:
        tuple: A tuple containing the padded short-term features, long-term features, labels, and masks.
    """

    short_term_features, long_term_features, short_term_masks, long_term_masks, current_short_frame, current_long_frame, labels = zip(*batch)

    # Get the lengths of each sequence
    short_term_lengths = [x.shape[0] for x in short_term_features]
    long_term_lengths = [x.shape[0] for x in long_term_features]

    max_short_term_len = max(short_term_lengths)
    max_long_term_len = max(long_term_lengths)

    # Create masks for ignoring padded elements
    short_term_padded = []
    short_term_masks = []
    for x, l in zip(short_term_features, short_term_lengths):
        padding = max_short_term_len - l
        short_term_padded.append(F.pad(x, (0, 0, 0, padding)))
        mask = [False] * l + [True] * padding
        short_term_masks.append(mask)

    long_term_padded = []
    long_term_masks = []
    for x, l in zip(long_term_features, long_term_lengths):
        padding = max_long_term_len - l
        long_term_padded.append(F.pad(x, (0, 0, 0, padding)))
        mask = [False] * l + [True] * padding
        long_term_masks.append(mask)

    # Convert to tensors
    short_term_batch = torch.stack(short_term_padded)
    long_term_batch = torch.stack(long_term_padded)
    short_term_masks = torch.tensor(short_term_masks)
    long_term_masks = torch.tensor(long_term_masks)

    labels_batch = torch.stack(labels)

    current_short_frame_batch = torch.tensor(current_short_frame)
    current_long_frame_batch = torch.tensor(current_long_frame)

    return (
        short_term_batch,
        long_term_batch,
        short_term_masks,
        long_term_masks,
        current_short_frame_batch,
        current_long_frame_batch,
        labels_batch
    )


def prepare_dataloaders(
        hdf5_file,
        prev_short_term_window,
        next_short_term_window,
        prev_long_term_window,
        next_long_term_window,
        batch_size
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
        next_long_term_window=next_long_term_window
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
        collate_fn=collate_fn,
        num_workers=num_workers,
        worker_init_fn=open_hdf5_file,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        worker_init_fn=open_hdf5_file,
        persistent_workers=True
    )
    
    return train_loader, val_loader


def main():
    # Parse arguments
    args = parse_args()

    # Prepare the dataset and dataloaders
    train_loader, val_loader = prepare_dataloaders(
        args.hdf5_file,
        args.prev_short_term_window,
        args.next_short_term_window,
        args.prev_long_term_window, 
        args.next_long_term_window, 
        args.batch_size
    )

    if args.resume_checkpoint:
        # Load the model from a checkpoint
        model = SpeechToExpressionModel.load_from_checkpoint(args.resume_checkpoint)

    else:
        # Initialize the model
        model = SpeechToExpressionModel(
            embed_dim=WAV2VEC2_EMBED_DIM,
            output_dim=FACIAL_FEATURE_DIM,
            num_heads=args.num_heads,
            num_steps=args.num_steps,
            num_layers=args.num_layers,
            lr=args.learning_rate
        )

    # Set up logging using TensorBoard
    log_dir = pathlib.Path(__file__).parent.parent / "logs"
    logger = TensorBoardLogger(log_dir.as_posix(), name="speech_to_expression")

    # Set up checkpoints to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", 
        filename="best_model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min"
    )

    # Set up early stopping if needed (optional)
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
