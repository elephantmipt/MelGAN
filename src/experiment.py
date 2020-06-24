from collections import OrderedDict

from src.data import MelFromDisk
from catalyst.dl.experiment import ConfigExperiment


class Experiment(ConfigExperiment):
    def get_datasets(
        self, stage: str, epoch: int = None, path:str = None, **kwargs,
    ) -> "OrderedDict[str, Dataset]":
        """
        Get dataset function in experiment defines logic
        of dataset loading from path. For more information
        you can read catalyst docs.
        Args:
            stage: stage name
            epoch: epoch idx
            path: path to wav and mel
            **kwargs: other staff

        Returns:
            Ordered dict with dataset
        """
        if path is None:
            path = "data/LJDpeech-1.1/wavs"  # you can use bin script
            # to download this dataset
        train_dataset = MelFromDisk(path)
        return OrderedDict({"train": train_dataset})
