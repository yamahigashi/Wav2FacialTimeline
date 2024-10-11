import os
import tempfile

import cv2
import librosa
import torch
import torch.nn.functional as F
import h5py
import pandas as pd
import numpy as np

import typing
if typing.TYPE_CHECKING:
    from feat import Detector
    from transformers import Wav2Vec2Processor, Wav2Vec2Model


def extract_facial_expression(file, feat_detector):
    # type: (str, Detector) -> pd.DataFrame
    """Extract facial expression data using py-feat."""

    frame_data = feat_detector.detect_video(file)
    scores = frame_data["FaceScore"]
    au_features = frame_data.aus
    emotions = frame_data.emotions
    poses = frame_data.poses
    poses = poses.diff().fillna(0)  # Calculate the difference between consecutive frames

    expression = pd.concat([scores, au_features, emotions, poses], axis=1)

    return expression


def is_video_reliable(video_file, feat_detector):
    # type: (str, Detector) -> bool
    """Check if the video is reliable by examining the face scores."""

    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return False

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_path = temp_file.name
        cv2.imwrite(temp_file.name, frame)

    try:
        frame_data = feat_detector.detect_image(temp_file.name)
        face_scores = frame_data["FaceScore"]
        yaw = frame_data.poses["Yaw"]

        if face_scores.empty:
            return False

        if face_scores.mean() < 0.97:
            return False

        if yaw.abs().mean() > 30:
            return False

    finally:
        os.unlink(temp_path)

    return True


def resample_data(data, target_length, columns=None):
    # type: (np.ndarray|torch.Tensor, int, list[str]|None) -> np.ndarray|pd.DataFrame
    """Resample the data to match the target frame rate."""

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    
    if data.ndim == 2:
        data = data.unsqueeze(0)
    else:
        data = data.unsqueeze(0).unsqueeze(0)
   
    resampled_data = F.interpolate(
        data.transpose(1, 2),
        size=target_length, 
        mode="linear", 
        align_corners=False
    ).transpose(1, 2)

    resampled_data = resampled_data.squeeze(0).numpy()

    if columns is not None:
        resampled_data = pd.DataFrame(resampled_data, columns=columns)

    return resampled_data


def process_video_fps(file):
    # type: (str) -> float
    """Retrieve the frames per second (FPS) of a video."""

    cap = cv2.VideoCapture(file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    return fps


def extract_audio_features(file, wav2vec_processor, wav2vec_model):
    # type: (str, Wav2Vec2Processor, Wav2Vec2Model) -> torch.Tensor
    """Extract audio features from video using Wav2Vec2."""

    audio, sr = librosa.load(file, sr=16000)
    inputs = wav2vec_processor(audio, return_tensors="pt", sampling_rate=sr).input_values
    with torch.no_grad():
        audio_features = wav2vec_model(inputs).last_hidden_state

    return audio_features


def prepare_audio_features_and_masks(
    audio_features,
    frame,
    prev_short_window,
    next_short_window,
    prev_long_window,
    next_long_window,
    embed_dim,
    device=None
):
    # type: (torch.Tensor, int, int, int, int, int, int, ...) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    """
    Prepare audio features and corresponding masks for short-term and long-term windows centered around a specific frame.

    This function extracts short-term and long-term audio feature windows centered at a given frame index. It also generates masks for these windows to handle padding or variable sequence lengths in batch processing. Additionally, it calculates the position of the current frame within each window.

    Args:
        audio_features (torch.Tensor): The full sequence of audio feature vectors with shape `(sequence_length, feature_dim)`.
        frame (int): The index of the current frame in the sequence.
        prev_short_window (int): Number of previous frames to include in the short-term window.
        next_short_window (int): Number of subsequent frames to include in the short-term window.
        prev_long_window (int): Number of previous frames to include in the long-term window.
        next_long_window (int): Number of subsequent frames to include in the long-term window.
        embed_dim (int): The dimensionality of the audio feature vectors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - **short_term_features** (`torch.Tensor`): Extracted short-term audio features with shape `(short_window_length, feature_dim)`.
            - **long_term_features** (`torch.Tensor`): Extracted long-term audio features with shape `(long_window_length, feature_dim)`.
            - **short_frame_mask** (`torch.Tensor`): Mask for the short-term window indicating valid positions.
            - **long_frame_mask** (`torch.Tensor`): Mask for the long-term window indicating valid positions.
            - **current_short_frame** (`torch.Tensor`): Index of the current frame within the short-term window.
            - **current_long_frame** (`torch.Tensor`): Index of the current frame within the long-term window.
    """
    audio_length = audio_features.shape[0]

    assert frame >= 0, "Frame index must be non-negative."
    assert prev_short_window >= 0, "Previous short-term window must be non-negative."
    assert next_short_window >= 0, "Next short-term window must be non-negative."
    assert prev_long_window >= 0, "Previous long-term window must be non-negative."
    assert next_long_window >= 0, "Next long-term window must be non-negative."

    st_start = min(prev_short_window, frame)
    lt_start = min(prev_long_window, frame)
    st_end   = min(next_short_window, audio_length - frame - 1)
    lt_end   = min(next_long_window, audio_length - frame - 1)

    # Define the range for short-term memory (past and future)
    st_start = max(0, frame - st_start)  # Past frames
    st_end = min(audio_length, frame + st_end + 1)  # Future frames
    st_prev_pad = abs(min(frame - prev_short_window, 0))
    st_next_pad = abs(max((frame + next_short_window + 1) - audio_length, 0))

    # Define the range for long-term memory (past and future)
    lt_start = max(0, frame - lt_start)  # Past frames
    lt_end = min(len(audio_features), frame + lt_end + 1)  # Future frames
    lt_prev_pad = abs(min(frame - prev_long_window, 0))
    lt_next_pad = abs(max((frame + next_long_window + 1) - audio_length, 0))

    # Get the short-term and long-term features
    st_features = audio_features[st_start:st_end]
    lt_frames = audio_features[lt_start:lt_end]

    # Convert to tensors
    st_features = torch.tensor(st_features, dtype=torch.float32, device=device)
    lt_features = torch.tensor(lt_frames, dtype=torch.float32, device=device)

    # Pad the short-term and long-term features
    st_features = F.pad(st_features, (0, 0, st_prev_pad, st_next_pad))
    lt_features = F.pad(lt_features, (0, 0, lt_prev_pad, lt_next_pad))

    # Create masks based on whether the frame is near the start or end of the sequence
    short_frame_mask = calculate_frame_masks(audio_length, frame, prev_short_window, next_short_window, device)
    long_frame_mask = calculate_frame_masks(audio_length, frame, prev_long_window, next_long_window, device)
    # 
    # if st_features.shape[0] > audio_length:
    #     st_features = st_features[:audio_length]
    # if lt_features.shape[0] > audio_length:
    #     lt_features = lt_features[:audio_length]
    # 
    # if long_frame_mask.shape[0] > audio_length:
    #     long_frame_mask = long_frame_mask[:audio_length]
    # if short_frame_mask.shape[0] > audio_length:
    #     short_frame_mask = short_frame_mask[:audio_length]

    current_short_frame = torch.tensor(max(0, frame - st_start), device=device)
    current_long_frame = torch.tensor(max(0, frame - lt_start), device=device)

    return st_features, lt_features, short_frame_mask, long_frame_mask, current_short_frame, current_long_frame


def calculate_frame_masks(total_length, current_frame, prev_window, next_window, device=None):
    # type: (int, int, int, int, torch.device|None) -> torch.Tensor
    """Calculate the frame masks for the given frame index."""

    prev_masks = torch.zeros(prev_window, dtype=torch.bool, device=device)
    prev_masks[:max(-prev_window, current_frame - prev_window)] = True

    next_masks = torch.zeros(next_window, dtype=torch.bool, device=device)
    next_masks[min(next_window, (total_length - current_frame)):] = True

    masks = torch.cat((
        prev_masks,
        torch.zeros(1, dtype=torch.bool, device=device),  # Current frame is always mask False
        next_masks
    ))

    return masks
