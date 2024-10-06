import os
import tempfile
import pathlib

import cv2
import librosa
import torch
import torch.nn.functional as F
import h5py
import pandas as pd
import numpy as np

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
