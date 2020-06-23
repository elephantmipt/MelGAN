from typing import Any, Mapping

from catalyst import dl, utils


class MelGANRunner(dl.Runner):
    """Runner"""

    def _handle_batch(self, batch: Mapping[str, Any]) -> None:
        model = utils.get_nn_from_ddp_module(self.model)
        generator = model["generator"]
        discriminator = model["discriminator"]

        generated_audio = generator(batch["generator_mel"])[
            :, :, : batch["generator_segment_len"]
        ]
        disc_fake = discriminator(generated_audio)  # probably slice here
        disc_real = discriminator(batch["generator_audio"])
        self.output = {"generator": {}, "discriminator": {}}
        self.output["generator"]["fake"] = disc_fake
        self.output["generator"]["real"] = disc_real
        generated_audio = generator(batch["discriminator_mel"])[
            :, :, : batch["discriminator_segment_len"]
        ]
        generated_audio = generated_audio.detach()
        disc_fake = discriminator(generated_audio)  # probably slice here
        disc_real = discriminator(batch["discriminator_audio"])
        self.output["discriminator"]["fake"] = disc_fake
        self.output["discriminator"]["real"] = disc_real
