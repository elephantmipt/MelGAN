from collections import OrderedDict

from catalyst import dl
from src.callbacks.discriminator_loss_callback import DiscriminatorLossCallback
from src.callbacks.generator_loss_callback import GeneratorLossCallback
from src.data.dataset import MelFromDisk
from src.models import Discriminator, Generator
from src.runner import MelGANRunner
import torch


def test():
    """Test Notebook API"""
    dataset = MelFromDisk(path="data/test")
    dataloader = torch.utils.data.DataLoader(dataset)
    loaders = OrderedDict({"train": dataloader})
    generator = Generator(80)
    discriminator = Discriminator()

    model = torch.nn.ModuleDict(
        {"generator": generator, "discriminator": discriminator}
    )
    optimizer = {
        "opt_g": torch.optim.Adam(generator.parameters()),
        "opt_d": torch.optim.Adam(discriminator.parameters()),
    }
    callbacks = {
        "loss_g": GeneratorLossCallback(),
        "loss_d": DiscriminatorLossCallback(),
        "o_g": dl.OptimizerCallback(
            metric_key="generator_loss", optimizer_key="opt_g"
        ),
        "o_d": dl.OptimizerCallback(
            metric_key="discriminator_loss", optimizer_key="opt_d"
        ),
    }
    runner = MelGANRunner()

    runner.train(
        model=model,
        loaders=loaders,
        optimizer=optimizer,
        callbacks=callbacks,
        check=True,
        main_metric="discriminator_loss",
    )
