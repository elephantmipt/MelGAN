from typing import List
import torch

from catalyst.dl import Callback, CallbackOrder
from scipy.io.wavfile import write
from catalyst.utils import get_nn_from_ddp_module

MAX_WAV_VALUE = 32768.0


class GenerateAudioCallback(Callback):
    def __init__(self, period: int = 10, mel_path: str = None, out_name: str = None):
        super().__init__(order=CallbackOrder.Internal)  # to set commit=False in wandb
        self.period = period
        self.mel_path = mel_path or "data/LJSpeech-1.1/wavs/LJ001-0006.mel"
        if out_name is None:
            out_name = "./reconstructed"
        self.out_name = out_name

    def on_epoch_end(self, runner: "IRunner"):
        if (runner.epoch - 1) % 10 == 0:
            mel = torch.load(self.mel_path)
            hop_length = 256
            # pad input mel with zeros to cut artifact
            # see https://github.com/seungwonpark/melgan/issues/8
            zero = torch.full((1, 80, 10), -11.5129).to(mel.device)
            mel = torch.cat((mel, zero), dim=2)
            generator = get_nn_from_ddp_module(runner.model)["generator"]
            if torch.cuda.is_available():
                mel.to("cuda")
                mel = mel.type(torch.cuda.FloatTensor)
            audio = generator.forward(mel).detach().cpu()
            audio = audio.squeeze()  # collapse all dimension except time axis
            audio = audio[:-(hop_length * 10)]
            audio = MAX_WAV_VALUE * audio
            audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE - 1)
            audio = audio.short()
            audio = audio.cpu().detach().numpy()
            try:
                import wandb
                wandb.log(
                    {
                        f"generated_{runner.epoch}.wav": [
                            wandb.Audio(audio,
                                        caption=self.mel_path,
                                        sample_rate=22050)
                        ]
                     }, step=runner.epoch
                )
            except:
                Warning("can't import wandb")
            out_path = self.out_name + f"_{runner.epoch}.wav"
            write(out_path, 22050, audio)
