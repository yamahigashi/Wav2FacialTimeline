import argparse

from tqdm import tqdm
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import pandas as pd

import utils
from model import SpeechToExpressionModel  # Assuming your model is saved here

import typing
if typing.TYPE_CHECKING:
    from  typing import Generator  # noqa: F401


########################################################################################

def parse_args():
    # type: () -> argparse.Namespace

    parser = argparse.ArgumentParser(description="Run inference on audio data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input audio file")
    parser.add_argument("--output_file", type=str, required=False, help="Path to save the output")

    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load the pre-trained model from a checkpoint and move it to the specified device."""
    model = SpeechToExpressionModel(
        embed_dim=768,
        output_dim=31,
        num_heads=8,
        num_steps=1010,
        num_layers=2,
        num_speakers=1,
        lr=1e-3
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)  # Move model to the device (GPU/CPU)

    return model


def preprocess_inference_data(audio_path, batch_size=1):
    # type: (str, int) -> Generator
    """Preprocess input data for inference with batching."""
    SHORT_TERM_WINDOW = 5
    LONG_TERM_WINDOW = 100

    wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    audio_features = utils.extract_audio_features(audio_path, wav2vec_processor, wav2vec_model)[0]
    num_frames = audio_features.shape[0]

    short_term_batch, long_term_batch = [], []

    for i in range(num_frames):
        if i < SHORT_TERM_WINDOW:
            short_term_features = audio_features[:i + 1]  # Padding for first frames
        else:
            short_term_features = audio_features[i - SHORT_TERM_WINDOW:i]

        long_term_start = max(0, i - LONG_TERM_WINDOW)
        long_term_frames = audio_features[long_term_start:i]

        if len(long_term_frames) == 0:
            long_term_frames = np.zeros((1, audio_features.shape[1]))

        short_term_batch.append(torch.tensor(short_term_features, dtype=torch.float32))
        long_term_batch.append(torch.tensor(long_term_frames, dtype=torch.float32))

        if len(short_term_batch) == batch_size:
            # Pad sequences to the same length
            padded_short_term_batch = pad_sequence(short_term_batch, batch_first=True)
            padded_long_term_batch = pad_sequence(long_term_batch, batch_first=True)

            yield padded_short_term_batch, padded_long_term_batch

            # Reset batch lists
            short_term_batch, long_term_batch = [], []

    # Yield any remaining data that didn't fill a complete batch
    if short_term_batch:
        padded_short_term_batch = pad_sequence(short_term_batch, batch_first=True)
        padded_long_term_batch = pad_sequence(long_term_batch, batch_first=True)
        yield padded_short_term_batch, padded_long_term_batch


def run_inference(model, frame_features, global_frame_features, device):
    # type: (torch.nn.Module, torch.Tensor, torch.Tensor, torch.device) -> torch.Tensor
    """Run inference on the given data."""

    frame_features = frame_features.to(device)
    global_frame_features = global_frame_features.to(device)

    with torch.no_grad():
        output = model(frame_features, global_frame_features)
    return output


def save_output(output, output_file_path):
    """Save inference output to a file or for further processing."""
    # Example: Save to a NumPy array or other format
    np_output = output.cpu().numpy()
    np.save(output_file_path, np_output)


def save_output_to_csv(results, output_file_path):
    """Save inference output to a CSV file using Pandas."""

    au_columns = [
        "AU01",
        "AU02",
        "AU04",
        "AU05",
        "AU06",
        "AU07",
        "AU09",
        "AU10",
        "AU11",
        "AU12",
        "AU14",
        "AU15",
        "AU17",
        "AU20",
        "AU23",
        "AU24",
        "AU25",
        "AU26",
        "AU28",
        "AU43",
    ]

    emotion_columns = [
        "Anger",
        "Disgust",
        "Fear",
        "Happiness",
        "Sadness",
        "Surprise",
        "Neutral",
    ]

    poses_columns = [
        "Pitch",
        "Roll",
        "Yaw",
    ]

    columns = ["FaceScore"] + au_columns + emotion_columns + poses_columns

    # Convert the list of tensors into a flat list and then a DataFrame
    flat_results = [np.round(tensor.numpy().flatten(), 3) for tensor in results]  # Flatten the tensors
    df = pd.DataFrame(flat_results, columns=columns)

    # Save the DataFrame to a CSV file
    df.to_csv(output_file_path, index=False)


def main():

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    model = load_model(args.checkpoint, device)

    data_gen = preprocess_inference_data(args.input_file)

    for frame_features, global_frame_features in tqdm(data_gen, desc="Processing frames"):
        output = run_inference(model, frame_features, global_frame_features, device)

        results.append(output.cpu())

    # Save or process the output
    if args.output_file:
        save_output_to_csv(results, args.output_file)

    else:
        for output in results:
            print(output)


if __name__ == "__main__":
    main()
