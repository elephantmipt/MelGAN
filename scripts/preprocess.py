import argparse
import glob
import os

import numpy as np
from src.preprocess.audio_to_mel import TacotronSTFT
from src.utils import read_wav_np
import torch
import tqdm


def main(args):
    stft = TacotronSTFT()

    wav_files = glob.glob(
        os.path.join(args.data_path, "**", "*.wav"), recursive=True
    )

    for wavpath in tqdm.tqdm(wav_files, desc="preprocess wav to mel"):
        sr, wav = read_wav_np(wavpath)
        assert sr == args.sampling_rate, (
            "sample rate mismatch. expected %d, got %d at %s"
            % (args.sampling_rate, sr, wavpath)
        )

        if len(wav) < args.segment_length + args.pad_short:
            wav = np.pad(
                wav,
                (0, args.segment_length + args.pad_short - len(wav)),
                mode="constant",
                constant_values=0.0,
            )

        wav = torch.from_numpy(wav).unsqueeze(0)
        mel = stft.mel_spectrogram(wav)

        melpath = wavpath.replace(".wav", ".mel")
        torch.save(mel, melpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-path",
        type=str,
        required=True,
        help="root directory of wav files",
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=16000,
        help="audio segment length for training",
    )
    parser.add_argument(
        "--sampling-rate", type=int, default=22050, help="audio rate"
    )
    parser.add_argument("--pad-short", type=int, default=2000)
    args = parser.parse_args()

    main(args)
