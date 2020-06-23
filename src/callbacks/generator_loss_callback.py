from typing import Any, Dict, List, Union

from catalyst import dl
import torch


class GeneratorLossCallback(dl.MetricCallback):
    def __init__(
        self,
        prefix: str = "generator_loss",
        output_key: Union[str, List[str], Dict[str, str]] = "generator",
        multiplier: float = 1.0,
        feature_weight: float = 10.0,
    ):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(
            prefix=prefix,
            metric_fn=self._loss,
            output_key=output_key,
            input_key="generator_segment_len",
            multiplier=multiplier,
        )
        self.feature_weight = feature_weight

    def _loss(self, output: Dict[str, Any], *args, **kwargs):
        real_d = output["real"]
        fake_d = output["fake"]
        loss = 0.0
        for (
            (_disc_name_r, disc_out_real),
            (_disc_name_f, disc_out_fake),
        ) in zip(real_d.items(), fake_d.items()):
            score_fake = disc_out_fake["score"]
            loss += torch.mean(
                torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2])
            )
            for features_fake, features_real in zip(
                disc_out_fake["features"].values(), disc_out_real["features"].values()
            ):
                loss += (
                    0.33
                    * self.feature_weight
                    * torch.mean(torch.abs(features_fake - features_real))
                )
        return loss
