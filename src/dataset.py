import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import utils

import typing
if typing.TYPE_CHECKING:
    Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]  # noqa: F401
    BatchData = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]  # noqa: F401


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

        # Open HDF5 file and retrieve all file names (keys) stored in the dataset
        self.hdf5_file = h5py.File(hdf5_file, "r")
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
        facial_expressions = self.hdf5_file[f"{file_key}/facial_expression"][:]
        audio_features = self.hdf5_file[f"{file_key}/audio_feature"][:]

        (
            short_term_features,
            long_term_features,
            short_frame_mask,
            long_frame_mask,
            current_short_frame,
            current_long_frame

        ) = utils.prepare_audio_features_and_masks(
            audio_features,
            file_frame_idx,
            self.prev_short_term_window,
            self.next_short_term_window,
            self.prev_long_term_window,
            self.next_long_term_window,
            self.embed_dim,
        )

        labels = torch.tensor(facial_expressions[file_frame_idx], dtype=torch.float32)  # Label for the current frame

        if file_frame_idx == 0:
            last_x = torch.zeros_like(labels)
        else:
            last_x = torch.tensor(facial_expressions[file_frame_idx - 1], dtype=torch.float32)

        return last_x, short_term_features, long_term_features, short_frame_mask, long_frame_mask, current_short_frame, current_long_frame, labels


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
