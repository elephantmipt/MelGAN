import librosa
import torch


class AudioDataset(torch.utils.data.Dataset):
    """Dataset for audio loading"""

    def __init__(self, path: str, lazy: bool = True):
        self.path = path
        file
