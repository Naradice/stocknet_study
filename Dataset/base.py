# from multiprocessing import Pool
import random
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Dataset(Dataset):
    version = 9

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
        index_sampler=None,
        split_ratio=0.8,
        indices=None,
    ):
        self.seed(seed)
        self.mm_params = {}
        data = df[columns]
        self.org_data = data
        min_length = [1]
        if processes is not None:
            for process in processes:
                data = process(data)
                min_length.append(process.get_minimum_required_length())
        self.processes = processes

        self._min_index = max(min_length) - 1
        if self._min_index < 0:
            self._min_index = 0
        if index_sampler is None:
            self.index_sampler = random_sampling
        elif type(index_sampler) is str and "k" in index_sampler:
            self.index_sampler = k_fold_sampling
        else:
            self.index_sampler = index_sampler

        self.observation_length = observation_length
        self.is_training = is_training
        self.device = device
        self._data = data
        self._columns = columns
        self._prediction_length = prediction_length
        if indices is None:
            self._init_indicies(data.index, randomize, split_ratio=split_ratio)
        else:
            self._init_indicies_row(indices, randomize, split_ratio=split_ratio)

    def _init_indicies(self, index, randomize=False, split_ratio=0.8):
        length = len(index) - self.observation_length - self._prediction_length
        if length <= 0:
            raise Exception(f"date length {length} is less than observation_length {self.observation_length}")

        self.train_indices, self.eval_indices = self.index_sampler(
            index, self._min_index, randomize, split_ratio, self.observation_length, self._prediction_length
        )

        if self.is_training:
            self._indices = self.train_indices
        else:
            self._indices = self.eval_indices

    def _init_indicies_row(self, index, randomize=False, split_ratio=0.8):
        length = len(index) - self.observation_length - self._prediction_length
        if length <= 0:
            raise Exception(f"date length {length} is less than observation_length {self.observation_length}")

        self.train_indices, self.eval_indices = random_sampling_row(
            index, self._min_index, randomize, split_ratio, self.observation_length, self._prediction_length
        )

        if self.is_training:
            self._indices = self.train_indices
        else:
            self._indices = self.eval_indices

    def output_indices(self, index):
        return slice(index + self.observation_length, index + self.observation_length + self._prediction_length)

    def _output_func(self, batch_size):
        if type(batch_size) == int:
            index = self._indices[batch_size]
            ndx = self.output_indices(index)
            ans = self._data[self._columns].iloc[ndx].values
            ans = torch.tensor(ans, device=self.device, dtype=torch.float)
            return ans
        elif type(batch_size) == slice:
            batch_indices = batch_size
            chunk_data = []
            for index in self._indices[batch_indices]:
                ndx = self.output_indices(index)
                chunk_data.append(self._data[self._columns].iloc[ndx].values.tolist())

            return torch.tensor(chunk_data, device=self.device, dtype=torch.float).transpose(0, 1)

    def input_indices(self, index):
        return slice(index, index + self.observation_length)

    def _input_func(self, batch_size):
        if type(batch_size) == int:
            index = self._indices[batch_size]
            ndx = self.input_indices(index)
            src = self._data[ndx].values
            src = torch.tensor(src, device=self.device, dtype=torch.float)
            return src
        elif type(batch_size) == slice:
            batch_indices = batch_size
            chunk_src = []
            for index in self._indices[batch_indices]:
                ndx = self.input_indices(index)
                chunk_src.append(self._data[self._columns].iloc[ndx].values.tolist())

            return torch.tensor(chunk_src, device=self.device, dtype=torch.float).transpose(0, 1)

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
        self._indices = random.sample(self.eval_indices, k=len(self.eval_indices))
        self.is_training = False

    def train(self):
        self._indices = random.sample(self.train_indices, k=len(self.train_indices))
        self.is_training = True

    def get_index_range(self):
        return min(self._indices), max(self._indices)

    def get_date_range(self):
        min_index, max_index = self.get_index_range()
        return self._data.index[min_index], self._data.index[max_index]

    def get_actual_index(self, ndx):
        inputs = []
        if type(ndx) == slice:
            inputs = self._indices[ndx]
        elif isinstance(ndx, Iterable):
            for index in ndx:
                inputs.append(self._indices[index])
        else:
            return self._indices[ndx]

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


class TimeDataset(Dataset):
    def __init__(
        self,
        df,
        columns: list,
        processes,
        time_column="index",
        observation_length: int = 60,
        device="cuda",
        prediction_length=10,
        seed=1017,
        is_training=True,
        randomize=True,
        index_sampler=None,
        indices=None,
    ):
        """return time data in addition to the columns data
        ((observation_length, CHUNK_SIZE, NUM_FEATURES), ((prediction_length, CHUNK_SIZE, 1)) as (feature_data, time_data) for source and target
        Args:
            df (pd.DataFrame): _description_
            columns (list): target columns like ["open", "high", "low", "close", "volume"]
            processes (list): list of process to add indicater and/or run standalization
            time_column (str, optional): specify column name or index. Defaults to "index"
            observation_length (int, optional): specify observation_length for source data. Defaults to 60.
            device (str, optional): Defaults to "cuda".
            prediction_length (int, optional): specify prediction_length for target data. Defaults to 10.
            seed (int, optional): specify random seed. Defaults to 1017.
            is_training (bool, optional): specify training mode or not. Defaults to True.
            randomize (bool, optional): specify randomize the index or not. Defaults to True.
        """
        if time_column == "index" and isinstance(df.index, pd.DatetimeIndex):
            self.time_column = "index"
        else:
            self.time_column = time_column
            columns += self.time_column

        super().__init__(
            df, columns, observation_length, device, processes, prediction_length, seed, is_training, randomize, index_sampler, indices=indices
        )

    def _output_func(self, batch_size):
        if type(batch_size) == int:
            index = self._indices[batch_size]
            ndx = self.output_indices(index)
            ans = self._data[self._columns].iloc[ndx].values.tolist()
            time = self._data[self.time_column].iloc[ndx].values.tolist()
            return ans, time
        elif type(batch_size) == slice:
            batch_indices = batch_size
            chunk_data = []
            time_chunk_data = []
            for index in self._indices[batch_indices]:
                ndx = self.output_indices(index)
                chunk_data.append(self._data[self._columns].iloc[ndx].values.tolist())
                time_chunk_data.append(self._data[self.time_column].iloc[ndx].values.tolist())

            return (
                torch.tensor(chunk_data, device=self.device, dtype=torch.float).transpose(0, 1),
                torch.tensor(time_chunk_data, device=self.device, dtype=torch.int).transpose(0, 1),
            )

    def _input_func(self, batch_size):
        if type(batch_size) == int:
            index = self._indices[batch_size]
            ndx = self.input_indices(index)
            src = self._data[ndx].values.tolist()
            time = self._data[self.time_column].iloc[ndx].values.tolist()
            return src, time
        elif type(batch_size) == slice:
            batch_indices = batch_size
            chunk_src = []
            time_chunk_data = []
            for index in self._indices[batch_indices]:
                ndx = self.input_indices(index)
                chunk_src.append(self._data[self._columns].iloc[ndx].values.tolist())
                time_chunk_data.append(self._data[self.time_column].iloc[ndx].values.tolist())

            return (
                torch.tensor(chunk_src, device=self.device, dtype=torch.float).transpose(0, 1),
                torch.tensor(time_chunk_data, device=self.device, dtype=torch.int).transpose(0, 1),
            )


def random_sampling(index, min_index, randomize, split_ratio, observation_length, prediction_length, params=None):
    length = len(index) - observation_length - prediction_length
    to_index = int(length * split_ratio)
    from_index = min_index
    train_indices = list(range(from_index, to_index))
    if randomize:
        train_indices = random.sample(train_indices, k=to_index - from_index)
    else:
        train_indices = train_indices

    # Note: If unique value exits in validation data only, validation loss would be grater than expected
    from_index = int(length * split_ratio) + observation_length + prediction_length
    to_index = length
    eval_indices = list(range(from_index, to_index))
    if randomize:
        eval_indices = random.sample(eval_indices, k=to_index - from_index)
    else:
        eval_indices = eval_indices
    return train_indices, eval_indices


def random_sampling_row(index, min_index, randomize, split_ratio, observation_length, prediction_length, params=None):
    length = len(index) - observation_length - prediction_length
    to_index = int(length * split_ratio)
    train_indices = index[:to_index]
    if randomize:
        train_indices = random.sample(train_indices, k=to_index)

    from_index = int(length * split_ratio) + observation_length + prediction_length

    eval_indices = index[from_index:]
    if randomize:
        eval_indices = random.sample(eval_indices, k=len(eval_indices))
    return train_indices, eval_indices


def k_fold_sampling(index, min_index, randomize, split_ratio, observation_length, prediction_length, params: dict = None):
    n = len(index)
    if params is None or "k" not in params:
        k = 100
    else:
        k = int(params["k"])

    if randomize:
        train_fold_index = random.sample(range(k), int(k * split_ratio))
    else:
        train_fold_index = list(range(int(k * split_ratio)))

    # create fold index
    split_idx = np.linspace(min_index, n, k + 1, dtype=int)

    train_idx = []
    val_idx = []
    for i in range(k):
        if i in train_fold_index:
            train_idx.extend(list(range(split_idx[i], split_idx[i + 1] - prediction_length - observation_length)))
        else:
            val_idx.extend(list(range(split_idx[i], split_idx[i + 1] - prediction_length - observation_length)))
    return train_idx, val_idx
