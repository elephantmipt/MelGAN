from typing import Any, Dict, List, Union

from catalyst import dl
import torch


class DiscriminatorLossCallback(dl.MetricCallback):
    def __init__(
        self,
        prefix: str = "discriminator_loss",
        output_key: Union[str, List[str], Dict[str, str]] = "discriminator",
        multiplier: float = 1.0,
        feature_weight: float = 10.0,
    ):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(
            prefix=prefix,
            metric_fn=self._loss,
            output_key=output_key,
            multiplier=multiplier,
        )
        self.feature_weight = feature_weight

    def _loss(self, output: Dict[str, Any]):
        real_d = output["real"]
        fake_d = output["fake"]
        loss = 0.0
        for (
            (_disc_name_r, disc_out_real),
            (_disc_name_f, disc_out_fake),
        ) in zip(real_d.items(), fake_d.items()):
            score_fake = disc_out_fake["score"]
            score_real = disc_out_real["score"]
            loss += torch.mean(
                torch.sum(torch.pow(score_real - 1.0, 2), dim=[1, 2])
            )
            loss += torch.mean(torch.sum(torch.pow(score_fake, 2), dim=[1, 2]))
        return loss
