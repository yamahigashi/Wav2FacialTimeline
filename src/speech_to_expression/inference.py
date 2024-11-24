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
    from config import SpeechToExpressionConfig  # noqa: F401


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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams = checkpoint["hyper_parameters"].copy()
    model = SpeechToExpressionModel(**hparams)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)
    return model, hparams


def preprocess_inference_data(audio_path, hparams):
    # type: (str, dict) -> Generator
    """Preprocess input data for inference with batching."""

    batch_size = 1  # Set the batch size to 1 for inference, for sequential processing for now

    wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    audio_features = utils.extract_audio_features(audio_path, wav2vec_processor, wav2vec_model)[0]
    target_audio_length = int(audio_features.shape[0] * 30 / 50)  # 50 fps is the default rate for Wav2Vec2
    audio_features = utils.resample_data(
            audio_features.detach().cpu().numpy(),
            target_length=target_audio_length)

    num_frames = audio_features.shape[0]

    s_term_batch = []
    l_term_batch = []
    s_mask_batch = []
    l_mask_batch = []
    s_frame_batch = []
    l_frame_batch = []

    def clear_batch(stb, ltb, smb, lmb, sfb, lfb):
        stb.clear()
        ltb.clear()
        smb.clear()
        lmb.clear()
        sfb.clear()
        lfb.clear()

    for i in range(num_frames):

        (
            s_term_features,
            l_term_frames,
            s_frame_masks,
            l_frame_masks,
            current_s_frame,
            current_l_frame

        ) = get_audio_feature_parameters(audio_features, i, hparams)

        s_term_batch.append(s_term_features)
        l_term_batch.append(l_term_frames)
        s_mask_batch.append(s_frame_masks)
        l_mask_batch.append(l_frame_masks)
        s_frame_batch.append(current_s_frame)
        l_frame_batch.append(current_l_frame)

        if len(s_term_batch) % batch_size == 0:
            # Pad sequences to the same length
            padded_s_term_batch = pad_sequence(s_term_batch, batch_first=True)
            padded_l_term_batch = pad_sequence(l_term_batch, batch_first=True)
            padded_s_mask_batch = pad_sequence(s_mask_batch, batch_first=True)
            padded_l_mask_batch = pad_sequence(l_mask_batch, batch_first=True)
            s_frame = torch.stack(s_frame_batch)
            l_frame = torch.stack(l_frame_batch)

            yield padded_s_term_batch, padded_l_term_batch, padded_s_mask_batch, padded_l_mask_batch, s_frame, l_frame

            # Reset batch lists
            clear_batch(s_term_batch, l_term_batch, s_mask_batch, l_mask_batch, s_frame_batch, l_frame_batch)

    # Yield any remaining data that didn't fill a complete batch
    if s_term_batch:
        padded_s_term_batch = pad_sequence(s_term_batch, batch_first=True)
        padded_l_term_batch = pad_sequence(l_term_batch, batch_first=True)
        padded_s_mask_batch = pad_sequence(s_mask_batch, batch_first=True)
        padded_l_mask_batch = pad_sequence(l_mask_batch, batch_first=True)

        s_frame = torch.stack(s_frame_batch)
        l_frame = torch.stack(l_frame_batch)

        yield padded_s_term_batch, padded_l_term_batch, padded_s_mask_batch, padded_l_mask_batch, s_frame, l_frame


def get_audio_feature_parameters(audio_features, frame, hparams):
    # type: (torch.Tensor, int, dict) -> ...

    _hparams = hparams.get("config", None)  # type: SpeechToExpressionConfig
    pst_window = _hparams.st.prev_window
    nst_window = _hparams.st.next_window
    plt_window = _hparams.lt.prev_window
    nlt_window = _hparams.lt.next_window
    embed_dim = _hparams.embed_dim

    return utils.prepare_audio_features_and_masks(
        audio_features,
        frame,
        pst_window,
        nst_window,
        plt_window,
        nlt_window,
        embed_dim,
    )


def run_inference(
        model,
        last_x,
        frame_features,
        global_frame_features,
        frame_masks,
        global_frame_masks,
        current_short_frame,
        current_long_frame,
        device,
        hparams
):
    """Run inference on the given data."""

    frame_features = frame_features.to(device)
    global_frame_features = global_frame_features.to(device)
    frame_masks = frame_masks.to(device)
    global_frame_masks = global_frame_masks.to(device)
    current_short_frame = current_short_frame.to(device)
    current_long_frame = current_long_frame.to(device)

    with torch.no_grad():
        output = model.generate(
            last_x,
            frame_features,
            global_frame_features,
            frame_masks,
            global_frame_masks,
            current_short_frame,
            current_long_frame,
        )
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
    model, hparams = load_model(args.checkpoint, device)

    data_gen = preprocess_inference_data(args.input_file, hparams)

    last_x = torch.zeros(1, 31).to(device)
    for batch in tqdm(data_gen, desc="Processing frames"):
        (
            frame_features,
            global_frame_features,
            frame_masks,
            global_frame_masks,
            current_short_frame,
            current_long_frame
        )= batch

        output = run_inference(
            model,
            last_x,
            frame_features,
            global_frame_features,
            frame_masks,
            global_frame_masks,
            current_short_frame,
            current_long_frame,
            device,
            hparams
        )

        last_x = torch.tanh(output)
        res = output.cpu()
        results.append(res)

    # Save or process the output
    if args.output_file:
        save_output_to_csv(results, args.output_file)

    else:
        for output in results:
            print(output)


if __name__ == "__main__":
    main()
