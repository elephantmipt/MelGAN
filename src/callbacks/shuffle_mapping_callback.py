from catalyst.dl import Callback, CallbackOrder


class ShuffleDatasetCallback(Callback):
    def __init__(self):
        super().__init__(order=CallbackOrder.External)

    def on_loader_start(self, runner: "IRunner"):
        runner.loaders["train"].dataset.shuffle_mapping()
