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


MEAN = np.array([
    9.9547e-01,  5.5478e-01,  3.5493e-01,  3.6727e-01,  3.6637e-01,
    2.3827e-01,  3.0750e-01,  2.4124e-01,  2.1665e-01,  5.0878e-01,
    1.9951e-01,  3.3968e-01,  4.1550e-01,  4.3008e-01,  3.7704e-01,
    4.0904e-01,  2.3289e-01,  7.1519e-01,  4.8964e-01,  1.8028e-01,
    2.3140e-01,  1.2828e-01,  6.0053e-02,  7.2398e-02,  7.9008e-02,
    9.7008e-02,  2.4460e-01,  3.1827e-01, -1.0843e-03, -2.2771e-05,
    1.5233e-04
]).astype(np.float32)

STD = np.array([
    0.0293, 0.1849, 0.1850, 0.2320, 0.1162, 0.1837, 0.4258, 0.1628, 0.2931,
    0.4567, 0.2299, 0.1891, 0.1888, 0.1085, 0.4687, 0.1591, 0.2064, 0.3505,
    0.2641, 0.1743, 0.2658, 0.2270, 0.1780, 0.1759, 0.1970, 0.1839, 0.3139,
    0.3375, 1.0733, 0.8360, 1.2791
]).astype(np.float32)


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
):
    # type: (torch.Tensor, int, int, int, int, int, int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
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
    st_features = torch.tensor(st_features, dtype=torch.float32)
    lt_features = torch.tensor(lt_frames, dtype=torch.float32)

    # Pad the short-term and long-term features
    st_features = F.pad(st_features, (0, 0, st_prev_pad, st_next_pad))
    lt_features = F.pad(lt_features, (0, 0, lt_prev_pad, lt_next_pad))

    # Create masks based on whether the frame is near the start or end of the sequence
    short_frame_mask = calculate_frame_masks(audio_length, frame, prev_short_window, next_short_window)
    long_frame_mask = calculate_frame_masks(audio_length, frame, prev_long_window, next_long_window)

    current_short_frame = torch.tensor(max(0, frame - st_start))
    current_long_frame = torch.tensor(max(0, frame - lt_start))

    return st_features, lt_features, short_frame_mask, long_frame_mask, current_short_frame, current_long_frame


def prepare_facial_features_and_masks(
    facial_features,
    frame,
    prev_window,
):
    # type: (torch.Tensor, int, int) -> tuple[torch.Tensor, torch.Tensor]
    """
    Prepare audio features and corresponding masks for short-term and long-term windows centered around a specific frame.

    This function extracts short-term and long-term audio feature windows centered at a given frame index. It also generates masks for these windows to handle padding or variable sequence lengths in batch processing. Additionally, it calculates the position of the current frame within each window.

    Args:
        facial_features (torch.Tensor): The full sequence of audio feature vectors with shape `(sequence_length, feature_dim)`.
        frame (int): The index of the current frame in the sequence.
        prev_window (int): Number of previous frames to include in the long-term window.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - **long_term_features** (`torch.Tensor`): Extracted long-term audio features with shape `(long_window_length, feature_dim)`.
            - **long_frame_mask** (`torch.Tensor`): Mask for the long-term window indicating valid positions.
    """
    facial_length = facial_features.shape[0]

    assert frame >= 0, "Frame index must be non-negative."
    assert prev_window >= 0, "Previous long-term window must be non-negative."

    lt_start = min(prev_window, frame)
    lt_end   = facial_length - frame - 1

    # Define the range for long-term memory (past and future)
    lt_start = max(0, frame - lt_start)  # Past frames
    lt_end = min(len(facial_features), frame + lt_end + 1)  # Future frames
    lt_prev_pad = abs(min(frame - prev_window, 0))

    # Get the short-term and long-term features
    lt_frames = facial_features[lt_start:lt_end]

    # Convert to tensors
    lt_features = torch.tensor(lt_frames, dtype=torch.float32)

    # Pad or truncate the long-term features
    lt_features = F.pad(lt_features, (0, 0, lt_prev_pad, 0))
    lt_features = lt_features[:prev_window + 1]  # past frames + current frame

    # Create masks based on whether the frame is near the start or end of the sequence
    long_frame_mask = calculate_frame_masks(facial_length, frame, prev_window, 0)
    # ensure the current frame and current frame - 1 are not masked
    long_frame_mask[-1] = False
    long_frame_mask[-2] = False

    return lt_features, long_frame_mask


def calculate_frame_masks(total_length, current_frame, prev_window, next_window):
    """
    現在フレームを中心としたウィンドウ(prev_window + 1 + next_windowフレーム分)に対するマスクを計算します。

    マスクはBoolの1次元テンソルを返し、ウィンドウ内の各フレームがデータ範囲内に存在しない場合はTrue(無効)、
    存在する場合はFalse(有効)となります。

    Args:
        total_length (int): 全フレーム数
        current_frame (int): 現在のフレームインデックス
        prev_window (int): 現在フレーム前に遡るフレーム数
        next_window (int): 現在フレーム後のフレーム数

    Returns:
        torch.Tensor: 形状 (prev_window+1+next_window,) のブール型テンソル
                      Trueはそのフレームが利用不可(マスクされるべき)であることを意味します。
    """
    # window_length = prev_window + 1 + next_window
    # ウィンドウ内フレームのインデックスを計算
    frame_indices = np.arange(current_frame - prev_window, current_frame + next_window + 1)

    # データ範囲外のインデックスはTrueでマスク
    mask = (frame_indices < 0) | (frame_indices >= total_length)

    return torch.tensor(mask, dtype=torch.bool)


def apply_mean_and_std_normalization(data):
    # type: (np.ndarray) -> np.ndarray
    """Apply mean and standard deviation normalization to the data."""

    data -= MEAN
    data /= (STD + 1e-8)

    return data


def reverse_mean_and_std_normalization(data):
    # type: (np.ndarray) -> np.ndarray
    """Reverse the mean and standard deviation normalization of the data."""

    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    data *= (STD + 1e-8)
    data += MEAN

    return data.astype(np.float32)


def find_closest_divisible_num_heads(embed_dim, target_num_heads, max_heads=16):
    # type: (int, int, int) -> int
    """
    embed_dimを割り切れる、target_num_headsに最も近いnum_headsを見つける関数。
    
    Args:
        embed_dim (int): 埋め込みの次元数
        target_num_heads (int): 目標のヘッド数
        max_heads (int): 最大のヘッド数の制限（デフォルトは16）

    Returns:
        int: target_num_headsに最も近い割り切れるnum_headsの値
    """
    # 上限を制限して、1 から max_heads の範囲で割り切れる num_heads を探す
    possible_heads = []
    for num_heads in range(1, max_heads + 1):
        if embed_dim % num_heads == 0:
            possible_heads.append(num_heads)

    # 最も target_num_heads に近い値を選択
    closest_num_heads = min(possible_heads, key=lambda x: abs(x - target_num_heads))
    
    return closest_num_heads


