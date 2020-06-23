from typing import Any, Mapping

from catalyst import dl, utils
import torch


class MelGANRunner(dl.Runner):
    """Runner"""

    def _handle_batch(self, batch: Mapping[str, Any]) -> None:
        model = utils.get_nn_from_ddp_module(self.model)
        generator = model["generator"]
        discriminator = model["discriminator"]
        generator_batch = batch["generator"]
        discriminator_batch = batch["discriminator"]
        generated_audio = generator(generator_batch["mel"])[
            :, :, : generator_batch["segment_len"]
        ]
        disc_fake = discriminator(generated_audio)  # probably slice here
        disc_real = discriminator(batch["audio_gen"])
        self.output = {"generator": {}, "discriminator": {}}
        self.output["generator"]["fake"] = disc_fake
        self.output["generator"]["real"] = disc_real
        generated_audio = generator(discriminator_batch["mel"])[
            :, :, : discriminator_batch["segment_len"]
        ]
        generated_audio = generated_audio.detach()
        disc_fake = discriminator(generated_audio)  # probably slice here
        disc_real = discriminator(batch["audio_disc"])
        self.output["discriminator"]["fake"] = disc_fake
        self.output["discriminator"]["real"] = disc_real