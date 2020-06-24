from src.callbacks import (
    DiscriminatorLossCallback,
    GeneratorLossCallback,
)
from src.models import Generator, Discriminator
from src.runner import MelGANRunner as Runner
from catalyst.dl import registry
from src.experiment import Experiment

registry.Model(Generator)
registry.Model(Discriminator)

registry.Callback(GeneratorLossCallback)
registry.Callback(DiscriminatorLossCallback)
