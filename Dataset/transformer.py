from .base import Dataset


class FeatureDataset(Dataset):
    """Dataset to return (DataLength, BATCH_SIZE, FEATURE_SIZE), here FEATURE_SIZE means column size specified
    dataset[index: index+batch_size] returns (src, tgt)
    Note that using Dataloader is a bit slower than directly using in my env.
    """

    def __init__(
        self,
        df,
        columns: list,
        observation_length: int = 60,
        device="cuda",
        processes=None,
        prediction_length=10,
        seed=1017,
        is_training=True,
        randomize=True,
    ):
        super().__init__(df, columns, observation_length, device, processes, prediction_length, seed, is_training, randomize)

    def _output_indices(self, index):
        return slice(index + self.observation_length - 1, index + self.observation_length + self._prediction_length)
