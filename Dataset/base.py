from multiprocessing import Pool
import random

import numpy as np
import torch


class DataFrameDataset:
    """
    Common Dataset for DataFrame source
    return src, tgt
    src: (observation_length, batch_size, feature_size)
    tgt: (prediction_length, batch_size, feature_size)

    feature_size depends on columns size
    """

    version = 8

    def __init__(
        self,
        df,
        observation_length: int = 30,
        prediction_length=1,
        device="cuda",
        initial_process=None,
        in_columns=None,
        out_columns=None,
        grouped_by_symbol=True,
        is_training=True,
        randomize=True,
        num_worker=1,
        seed=1017,
    ):
        self.seed(seed)
        self.num_worker = num_worker
        self.mm_params = {}
        for process in initial_process:
            df = process.run(df)
        self.observation_length = observation_length
        self.predition_length = prediction_length
        self.is_training = is_training
        self.device = device
        self._data = df

        self.in_columns = df.columns
        if in_columns:
            self._columns = df.columns
        self.out_columns = df.columns
        if out_columns:
            self.out_columns = out_columns
        self._init_indicies(df, randomize)

    def _init_indicies(self, data, randomize=False, split_ratio=0.7):
        length = len(data) - self.observation_length - self.predition_length
        if length <= 0:
            raise Exception(
                f"date length {length} is less than observation_length {self.observation_length}"
            )

        # stopped from v7. As input data inluceds target data
        # indices = random.sample(range(1, length), k=length-1)

        from_index = 1
        to_index = int(length * split_ratio)
        train_indices = list(range(from_index, to_index))
        if randomize:
            self.train_indices = random.sample(train_indices, k=to_index - from_index)
        else:
            self.train_indices = train_indices

        # Note: If unique value exits in validation data only, validation loss would be grater than expected
        from_index = (
            int(length * split_ratio) + self.observation_length + self.predition_length
        )
        to_index = length
        eval_indices = list(range(from_index, to_index))
        if randomize:
            self.eval_indices = random.sample(eval_indices, k=to_index - from_index)
        else:
            self.eval_indices = eval_indices

        if self.is_training:
            self._indices = self.train_indices
        else:
            self._indices = self.eval_indices

    def __get_single_output(self, index):
        return (
            self._data[self.out_columns]
            .iloc[
                index
                + self.observation_length : index
                + self.observation_length
                + self.predition_length
            ]
            .values.tolist()
        )

    def _output_func(self, batch_size):
        if type(batch_size) == int:
            index = self._indices[batch_size]
            ans = self.__get_single_output(index)
            ans = torch.tensor(ans, device=self.device, dtype=torch.float)
            ans = ans.reshape(len(self.out_columns), 1, self.predition_length + 1)
            return ans.transpose(0, 2)
        elif type(batch_size) == slice:
            batch_indices = batch_size
            chunk_data = []
            with Pool(processes=4) as p:
                chunk_data = p.map(
                    self.__get_single_output, self._indices[batch_indices]
                )

            return torch.tensor(chunk_data, device=self.device, dtype=torch.float)

    def __get_single_input(self, index):
        return self._data[self.in_columns][
            index : index + self.observation_length
        ].values.tolist()

    def _input_func(self, batch_size):
        if type(batch_size) == int:
            index = self._indices[batch_size]
            src = self.__get_single_input(index)
            src = torch.tensor(src, device=self.device, dtype=torch.float)
            src = src.reshape(len(self._columns), 1, self.observation_length)
            return src
        elif type(batch_size) == slice:
            chunk_src = []
            with Pool(processes=4) as p:
                chunk_src = p.map(self.__get_single_input, self._indices[batch_size])

            return torch.tensor(chunk_src, device=self.device, dtype=torch.float)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, ndx):
        return self._input_func(ndx), self._output_func(ndx)

    def seed(self, seed=None):
        """ """
        if seed is None:
            seed = 1017
        else:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.seed_value = seed

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def render(self, mode="human", close=False):
        """ """
        pass

    def eval(self):
        self._indices = self.eval_indices
        self.is_training = False

    def train(self):
        self._indices = self.train_indices
        self.is_training = True

    def get_actual_index(self, ndx):
        inputs = []
        if type(ndx) == slice:
            for index in self._indices[ndx]:
                inputs.append(index)
        else:
            inputs = self._indices[ndx]
        return inputs

    def get_row_data(self, ndx):
        inputs = []
        if type(ndx) == slice:
            for index in self._indices[ndx]:
                df = self._data[index : index + self.observation_length]
                inputs.append(df)
        else:
            index = ndx
            inputs = df = self._data[index : index + self.observation_length]
        return inputs
