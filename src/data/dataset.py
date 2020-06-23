import glob
import os
import random

import numpy as np
from src.utils import read_wav_np
import torch
from torch.utils.data import DataLoader, Dataset


def create_dataloader(hp, args, train):
    dataset = MelFromDisk(hp, args, train)

    if train:
        return DataLoader(
            dataset=dataset,
            batch_size=hp.train.batch_size,
            shuffle=True,
            num_workers=hp.train.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    return DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=hp.train.num_workers,
        pin_memory=True,
        drop_last=False,
    )


class MelFromDisk(Dataset):
    def __init__(
            self,
            path: str,
            segment_length: int = 16000,
            pad_short: int = 2000,
            filter_length: int = 1024,
            hop_length: int = 256,
            train: bool = True
    ):

        self.train = train
        self.path = path
        self.segment_length = segment_length
        self.pad_short = pad_short
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.wav_list = glob.glob(
            os.path.join(self.path, "**", "*.wav"), recursive=True
        )
        self.mel_segment_length = (
                self.segment_length // self.hop_length + 2
        )
        self.mapping = [i for i in range(len(self.wav_list))]

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        if self.train:
            idx1 = idx
            idx2 = self.mapping[idx1]
            output_dict = {}
            for k, v in self.my_getitem(idx1).items():
                output_dict[f"generator_{k}"] = v
            for k, v in self.my_getitem(idx2).items():
                output_dict[f"discriminator_{k}"] = v

            return output_dict

        return self.my_getitem(idx)

    def shuffle_mapping(self):
        random.shuffle(self.mapping)

    def my_getitem(self, idx):
        wavpath = self.wav_list[idx]
        melpath = wavpath.replace(".wav", ".mel")
        sr, audio = read_wav_np(wavpath)
        if len(audio) < self.segment_length + self.pad_short:
            audio = np.pad(
                audio,
                (
                    0,
                    self.segment_length
                    + self.pad_short
                    - len(audio),
                ),
                mode="constant",
                constant_values=0.0,
            )

        audio = torch.from_numpy(audio).unsqueeze(0)
        mel = torch.load(melpath).squeeze(0)

        if self.train:
            max_mel_start = mel.size(1) - self.mel_segment_length
            mel_start = random.randint(0, max_mel_start)
            mel_end = mel_start + self.mel_segment_length
            mel = mel[:, mel_start:mel_end]

            audio_start = mel_start * self.hop_length
            audio = audio[
                    :, audio_start: audio_start + self.segment_length
                    ]

        audio = audio + (1 / 32768) * torch.randn_like(audio)
        return {
            "mel": mel,
            "audio": audio,
            "segment_len": self.segment_length,
        }
