import pathlib
import argparse
import random

import cv2
import librosa
import torch
import torch.nn.functional as F
import h5py
import pandas as pd
import numpy as np

from feat import Detector
from transformers import Wav2Vec2Processor, Wav2Vec2Model

import utils


def parse_args():
    # type: () -> argparse.Namespace

    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("--input_dir", type=str, help="Path to the input directory containing video files")
    parser.add_argument("--output_file", type=str, help="Path to the output HDF5 file")
    parser.add_argument("--frame_rate", type=int, default=30, help="Frame rate for the output video")
    parser.add_argument("--audio_dir", type=str, help="Path to the directory containing audio files")
    parser.add_argument("--repair", action="store_true", help="Check the contents of the HDF5 file", default=False)

    return parser.parse_args()


def store_datasets(hdf5_file, file_path, expression_data, audio_data):
    # Type: (h5py.File, pathlib.Path, np.ndarray, np.ndarray) -> None
    """Store individual datasets for each file in the HDF5 file.

    Args:
        hdf5_file (h5py.File): The HDF5 file object.
        file_name (pathlib.Path): The Path object for the file being processed.
        expression_data (np.array): The facial expression data to store.
        audio_data (np.array): The audio features to store.
    """
    # Use a unique dataset name for each file
    file_key = file_path.as_posix().replace("/", "_")
    hdf5_file.create_dataset(f"{file_key}/facial_expression", data=expression_data)
    hdf5_file.create_dataset(f"{file_key}/audio_feature", data=audio_data)


def find_audio_file(video_file_relative, audio_dir):
    # type: (pathlib.Path, str) -> pathlib.Path|None
    """Find the corresponding audio file for a given video file."""

    audio_exts = [".wav", ".aac", ".mp3"]

    audio_path = pathlib.Path(audio_dir) / video_file_relative
    for ext in audio_exts:
        audio_file = audio_path.with_suffix(ext)
        if audio_file.exists():
            return audio_file

    return None


def preprocess(movie_dir, output_file, frame_rate, audio_dir=None):
    # type: (str, str, int, str|None) -> None
    """Main preprocessing function for the RAVDESS dataset."""

    # Initialize Wav2Vec2 and py-feat detector
    wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    feat_detector = Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model="resmasknet",
        facepose_model="img2pose",
        # device="cpu",
        device="cuda",
    )

    # Create HDF5 file for storing features
    with h5py.File(output_file, "a") as hdf5_file:

        try:
            file_names = list(hdf5_file.keys())
        except (KeyError, FileNotFoundError):
            file_names = []

        # for idx, file in enumerate(pathlib.Path(movie_dir).rglob("01-*.mp4")):
        # files = list(pathlib.Path(movie_dir).rglob("**/front/**/*.mp4"))
        files = list(pathlib.Path(movie_dir).rglob("*.mp4"))
        random.shuffle(files)
        for idx, file in enumerate(files):
            file_key = file.as_posix().replace("/", "_")
            if file_key in file_names:
                print(f"Skipping {file}")
                continue

            if audio_dir is not None:
                audio_file = find_audio_file(file.relative_to(movie_dir), audio_dir)
                if audio_file is None:
                    print(f"Could not find audio file for {file}")
                    continue
            else:
                audio_file = file

            # ----------------------------------------
            # Video preprocessing (Facial expressions)
            # ----------------------------------------
            if not utils.is_video_reliable(file.as_posix(), feat_detector):
                print(f"Skipping {file} due to unreliable video")
                continue

            expression = utils.extract_facial_expression(file.as_posix(), feat_detector)
            if expression["FaceScore"].mean() < 0.97:
                print(f"Skipping {file} due to low face detection score")
                continue

            video_fps = utils.process_video_fps(file.as_posix())
            target_expression_length = int(expression.shape[0] * frame_rate / video_fps)

            # ----------------------------------------
            # Audio preprocessing (Wav2Vec2 features)
            # ----------------------------------------
            audio_features = utils.extract_audio_features(audio_file.as_posix(), wav2vec_processor, wav2vec_model)[0]
            target_audio_length = int(audio_features.shape[0] * frame_rate / 50)  # 50 fps is the default rate for Wav2Vec2
            target_length = min(target_expression_length, target_audio_length)

            downsampled_audio = utils.resample_data(
                audio_features.detach().cpu().numpy(),
                target_length=target_length)

            expression = utils.resample_data(
                expression.values,
                target_length=target_length,
                columns=expression.columns)

            # ----------------------------------------
            store_datasets(hdf5_file, file, expression.values, downsampled_audio)
            hdf5_file.flush()

    print(f"Preprocessing complete. Data saved to {output_file}")


def check_hdf5_file():
    args = parse_args()
    file_path = args.output_file

    with h5py.File(file_path, "r") as hdf5_file:
        print("Datasets in the file:")
        for i, dataset_name in enumerate(hdf5_file):
            print(f"- {dataset_name}: {hdf5_file[dataset_name]}")
            expressions = hdf5_file[dataset_name]["facial_expression"]
            print(np.round(expressions, 2)[:5])
            # print(f"""  - {hdf5_file[dataset_name]["facial_expression"]}""")
            # for j, group_name in enumerate(hdf5_file[dataset_name]):
            #     print(f"  - {group_name}: {hdf5_file[dataset_name][group_name]}")
            break


def repair_nan_inf_to_minus_one(file_path):
    # type: (str) -> None
    """Replace NaN and infinite values with -1 in the HDF5 file. And
    set the FaceScore to 0 if has NaN or infinite values.
    """

    def replace_nan_inf(data):
        # type: (np.ndarray) -> np.ndarray
        """Replace NaN and infinite values with -1."""
        data[~np.isfinite(data)] = -1
        return data

    with h5py.File(file_path, "a") as hdf5_file:
        for i, dataset_name in enumerate(hdf5_file):
            expressions = hdf5_file[dataset_name]["facial_expression"][:]
            audio_features = hdf5_file[dataset_name]["audio_feature"][:]

            if not torch.isfinite(torch.tensor(expressions)).all():
                print(f"Found NaN or infinite expressions values in {dataset_name}")
                expressions = replace_nan_inf(expressions)
                expressions[0] = 0  # [0] means FaceScore, set to 0
                hdf5_file[dataset_name]["facial_expression"][:] = expressions

            if not torch.isfinite(torch.tensor(audio_features)).all():
                print(f"Found NaN or infinite audio features in {dataset_name}")
                audio_features = replace_nan_inf(audio_features)
                expressions[0] = 0
                hdf5_file[dataset_name]["audio_feature"][:] = audio_features
                hdf5_file[dataset_name]["facial_expression"][:] = expressions


def main():
    args = parse_args()

    if args.repair:
        repair_nan_inf_to_minus_one(args.output_file)
    else:
        preprocess(args.input_dir, args.output_file, args.frame_rate, args.audio_dir)


if __name__ == "__main__":
    main()
    # check_hdf5_file()
