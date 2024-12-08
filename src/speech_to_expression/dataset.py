import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import utils

import typing
if typing.TYPE_CHECKING:
    from jaxtyping import (
        Float,  # noqa: F401
        Array,  # noqa: F401
        Int,  # noqa: F401
    )
    PastFrames = Float[Array, "seq_len", "output_dim"]
    Feat = Float[Array, "seq_len", "embed_dim"]
    Mask = Float[Array, "seq_len"]

    LabelBatch = Float[Array, "batch_size", "output_dim"]
    PastFramesBatch = Float[Array, "batch_size", "seq_len", "output_dim"]
    FeatBatch = Float[Array, "batch_size", "seq_len", "embed_dim"]
    MaskBatch = Float[Array, "batch_size", "seq_len"]
    Batch = tuple[PastFramesBatch, FeatBatch, FeatBatch, MaskBatch, MaskBatch, MaskBatch, Int, Int]
    BatchData = tuple[PastFrames, Feat, Feat, Mask, Mask, Mask, Int, Int]  # noqa: F401


class SpeakerDataset(Dataset):
    """Speaker dataset for training the model from HDF5 file.

    This dataset class reads the audio features and facial expressions from an HDF5 file
    and provides the short-term features, long-term features, and labels for training the model.
    """

    def __init__(
            self,
            hdf5_file,
            embed_dim,
            prev_short_term_window=4,
            next_short_term_window=3,
            prev_long_term_window=90,
            next_long_term_window=60,
    ):
        """Initialize the dataset by reading from the HDF5 file.

        Args:
            hdf5_file (str): Path to the HDF5 file containing audio features and facial expressions.
            embed_dim (int): Embedding dimension for the Transformer encoder.
            prev_short_term_window (int): Number of frames to consider in short-term memory.
            next_short_term_window (int): Number of frames to consider in short-term memory.
            prev_long_term_window (int): Number of frames to consider in long-term memory.
            next_long_term_window (int): Number of frames to consider in long-term memory.
        """
        self.hdf5_file_path = hdf5_file
        self.embed_dim = embed_dim
        self.prev_short_term_window = prev_short_term_window
        self.next_short_term_window = next_short_term_window
        self.prev_long_term_window = prev_long_term_window
        self.next_long_term_window = next_long_term_window

        self.hdf5_file = None
        self._initialize_file_offsets()

    def _initialize_file_offsets(self):
        """Initialize file offsets for all files in the dataset."""
        with h5py.File(self.hdf5_file_path, "r") as f:
            self.file_keys = list(f.keys())
            self.file_frame_offsets = []
            total_frames = 0

            for file_key in self.file_keys:
                num_frames = f[f"{file_key}/facial_expression"].shape[0]
                self.file_frame_offsets.append((file_key, total_frames, total_frames + num_frames))
                total_frames += num_frames

        self.total_frames = total_frames

    def __len__(self):
        """Return the total number of frames across all files."""
        return self.total_frames

    def _get_file_index(self, frame_idx):
        # type: (int) -> tuple[str, int]
        """Helper function to find which file a given frame index belongs to.

        Args:
            frame_idx (int): The frame index to find the corresponding file for.

        Returns:
            tuple: A tuple containing the file key and the frame index within that file.
        """

        for file_key, start_frame, end_frame in self.file_frame_offsets:
            if start_frame <= frame_idx < end_frame:
                return file_key, frame_idx - start_frame

        raise IndexError("Frame index out of range")

    def __getitem__(self, idx):
        # type: (int) -> BatchData
        """Retrieve a data sample for the given frame index."""

        if self.hdf5_file is None:
            raise RuntimeError("HDF5 file is not opened. Ensure `worker_init_fn` is used to open the file in each worker.")

        file_key, file_frame_idx = self._get_file_index(idx)

        # Get the facial expression window width
        facial_start_frame = max(0, file_frame_idx - self.prev_long_term_window)
        facial_end_frame = file_frame_idx + 1
        facial_expressions = self.hdf5_file[f"{file_key}/facial_expression"][facial_start_frame:facial_end_frame]

        # Get the audio feature window width
        audio_start_frame = max(0, file_frame_idx - max(self.prev_short_term_window, self.prev_long_term_window))
        audio_end_frame = min(
            self.hdf5_file[f"{file_key}/audio_feature"].shape[0], 
            file_frame_idx + max(self.next_short_term_window, self.next_long_term_window) + 1
        )
        audio_features = self.hdf5_file[f"{file_key}/audio_feature"][audio_start_frame:audio_end_frame]

        relative_audio_frame_idx = file_frame_idx - audio_start_frame
        relative_facial_frame_idx = file_frame_idx - facial_start_frame

        (
            short_term_features,
            long_term_features,
            short_frame_mask,
            long_frame_mask,
            current_short_frame,
            current_long_frame

        ) = utils.prepare_audio_features_and_masks(
            audio_features,
            relative_audio_frame_idx,
            self.prev_short_term_window,
            self.next_short_term_window,
            self.prev_long_term_window,
            self.next_long_term_window,
            self.embed_dim,
        )

        past_frames, past_frames_mask = utils.prepare_facial_features_and_masks(
            facial_expressions,
            relative_facial_frame_idx,
            self.prev_long_term_window
        )

        past_frames = utils.apply_mean_and_std_normalization(past_frames)

        return (
            past_frames,
            short_term_features,
            long_term_features,
            past_frames_mask,
            short_frame_mask,
            long_frame_mask,
            current_short_frame,
            current_long_frame
        )


def open_hdf5_file(_worker_id):
    """Initialize the worker process by opening the HDF5 file.

    Using this function as the `worker_init_fn` ensures that each worker opens the file
    in the DataLoader initialization process.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset

    if isinstance(dataset, SpeakerDataset):
        dataset.hdf5_file = h5py.File(dataset.hdf5_file_path, "r")
