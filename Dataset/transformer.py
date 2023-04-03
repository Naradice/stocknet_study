from .base import Dataset, TimeDataset


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
        index_sampler=None
    ):
        super().__init__(df, columns, observation_length, device, processes, prediction_length, seed, is_training, randomize,index_sampler)

    def output_indices(self, index):
        return slice(index + self.observation_length - 1, index + self.observation_length + self._prediction_length)


class TimeFeatureDataset(TimeDataset):
    """Dataset to return (DataLength, BATCH_SIZE, FEATURE_SIZE), here FEATURE_SIZE means column size specified
    dataset[index: index+batch_size] returns (src, tgt)
    Note that using Dataloader is a bit slower than directly using in my env.
    """

    def __init__(
        self,
        df,
        columns: list,
        time_column: str,
        observation_length: int = 60,
        device="cuda",
        processes=None,
        prediction_length=10,
        seed=1017,
        is_training=True,
        randomize=True,
        index_sampler=None
    ):
        super().__init__(
            df,
            columns,
            time_column=time_column,
            processes=processes,
            observation_length=observation_length,
            device=device,
            prediction_length=prediction_length,
            seed=seed,
            is_training=is_training,
            randomize=randomize,
            index_sampler=None
        )

    def output_indices(self, index):
        return slice(index + self.observation_length - 1, index + self.observation_length + self._prediction_length)
