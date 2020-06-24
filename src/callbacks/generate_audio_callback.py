from typing import List
import torch

from catalyst.dl import Callback, CallbackOrder
from scipy.io.wavfile import write
from catalyst.utils import get_nn_from_ddp_module

MAX_WAV_VALUE = 32768.0


class GenerateAudioCallback(Callback):
    def __init__(self, epochs: List[int] = None, mel_path: str = None, out_path: str = None):
        super().__init__(order=CallbackOrder.External)
        if epochs is None:
            epochs = [1, 10, 50, 100]
        self.epochs = epochs
        self.mel_path = mel_path or "data/LJSpeech-1.1/wavs/LJ001-0006.mel"
        if out_path is None:
            out_path = "./reconstructed"
        self.out_path = out_path

    def on_epoch_end(self, runner: "IRunner"):
        if runner.epoch in self.epochs:
            mel = torch.load(self.mel_path)
            hop_length = 256
            # pad input mel with zeros to cut artifact
            # see https://github.com/seungwonpark/melgan/issues/8
            zero = torch.full((1, 80, 10), -11.5129).to(mel.device)
            mel = torch.cat((mel, zero), dim=2)
            generator = get_nn_from_ddp_module(runner.model)["generator"]
            audio = generator.forward(mel)
            audio = audio.squeeze()  # collapse all dimension except time axis
            audio = audio[:-(hop_length * 10)]
            audio = MAX_WAV_VALUE * audio
            audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE - 1)
            audio = audio.short()
            audio = audio.cpu().detach().numpy()
            out_path = self.out_path + f"/{runner.epoch}.wav"
            write(out_path, 22050, audio)