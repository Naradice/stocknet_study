# from multiprocessing import Pool
import os
import random
import sys
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch

module_path = os.path.abspath("../fprocess")
sys.path.append(module_path)

from fprocess import fprocess


class Dataset:
    version = 8

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
        self.seed(seed)
        self.mm_params = {}
        data = df[columns]
        self.org_data = data
        min_length = [1]
        for process in processes:
            data = process(data)
            min_length.append(process.get_minimum_required_length())
        self.processes = processes

        self._min_index = max(min_length) - 1
        if self._min_index < 0:
            self._min_index = 0

        self.observation_length = observation_length
        self.is_training = is_training
        self.device = device
        self._data = data
        self._columns = columns
        self._prediction_length = prediction_length
        self._init_indicies(data, randomize)

    def _init_indicies(self, data, randomize=False, split_ratio=0.7):
        length = len(data) - self.observation_length - self._prediction_length
        if length <= 0:
            raise Exception(f"date length {length} is less than observation_length {self.observation_length}")

        # stopped from v7. As input data inluceds target data
        # indices = random.sample(range(1, length), k=length-1)

        from_index = self._min_index
        to_index = int(length * split_ratio)
        train_indices = list(range(from_index, to_index))
        if randomize:
            self.train_indices = random.sample(train_indices, k=to_index - from_index)
        else:
            self.train_indices = train_indices

        # Note: If unique value exits in validation data only, validation loss would be grater than expected
        from_index = int(length * split_ratio) + self.observation_length + self._prediction_length
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

    def _output_indices(self, index):
        return slice(index + self.observation_length, index + self.observation_length + self._prediction_length)

    def _output_func(self, batch_size):
        if type(batch_size) == int:
            index = self._indices[batch_size]
            ndx = self._output_indices(index)
            ans = self._data[self._columns].iloc[ndx].values.tolist()
            ans = ans.reshape(self._prediction_length + 1, 1, len(self._columns))
            return ans
        elif type(batch_size) == slice:
            batch_indices = batch_size
            chunk_data = []
            for index in self._indices[batch_indices]:
                ndx = self._output_indices(index)
                chunk_data.append(self._data[self._columns].iloc[ndx].values.tolist())

            return torch.tensor(chunk_data, device=self.device, dtype=torch.float).transpose(0, 1)

    def _input_indices(self, index):
        return slice(index, index + self.observation_length)

    def _input_func(self, batch_size):
        if type(batch_size) == int:
            index = self._indices[batch_size]
            ndx = self._input_indices(index)
            src = self._data[ndx].values.tolist()
            src = torch.tensor(src, device=self.device, dtype=torch.float)
            src = src.reshape(self.observation_length, 1, len(self._columns))
            return src
        elif type(batch_size) == slice:
            batch_indices = batch_size
            chunk_src = []
            for index in self._indices[batch_indices]:
                ndx = self._input_indices(index)
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
        self._indices = self.eval_indices
        self.is_training = False

    def train(self):
        self._indices = self.train_indices
        self.is_training = True

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

    def revert(self, values, ndx, is_tgt=False, columns=None):
        r_data = values
        indices = self.get_actual_index(ndx)
        if is_tgt:
            tgt_indices = []
            for __index in indices:
                ndx = self._output_indices(__index)
                tgt_indices.append(ndx.start)
            indices = tgt_indices
        # print(f"start revert procress for {[__process.kinds for __process in self.processes]}")
        for p_index in range(len(self.processes)):
            r_index = len(self.processes) - 1 - p_index
            process = self.processes[r_index]
            if hasattr(process, "revert_params"):
                # print(f"currently: {r_data[0, 0]}")
                params = process.revert_params
                if len(params) == 1:
                    r_data = process.revert(r_data)
                else:
                    params = {}
                    if process.kinds == fprocess.MinMaxPreProcess.kinds:
                        r_data = process.revert(r_data, columns=columns)
                    elif process.kinds == fprocess.SimpleColumnDiffPreProcess.kinds:
                        close_column = process.base_column
                        if p_index > 0:
                            processes = self.processes[:p_index]
                            required_length = [1]
                            base_processes = []
                            for base_process in processes:
                                if close_column in base_process.columns:
                                    base_processes.append(base_process)
                                    required_length.append(base_process.get_minimum_required_length())
                            if len(base_processes) > 0:
                                raise Exception("Not implemented yet")
                        base_indices = [index - 1 for index in indices]
                        base_values = self.org_data[close_column].iloc[base_indices]
                        r_data = process.revert(r_data, base_value=base_values)
                    elif process.kinds == fprocess.DiffPreProcess.kinds:
                        if columns is None:
                            target_columns = process.columns
                        else:
                            target_columns = columns
                        if r_index > 0:
                            processes = self.processes[:r_index]
                            required_length = [process.get_minimum_required_length()]
                            base_processes = []
                            for base_process in processes:
                                if len(set(target_columns) & set(base_process.columns)) > 0:
                                    base_processes.append(base_process)
                                    required_length.append(base_process.get_minimum_required_length())
                            if len(base_processes) > 0:
                                required_length = max(required_length)
                                batch_base_indices = [index - required_length for index in indices]
                                batch_base_values = pd.DataFrame()
                                #print(f"  apply {[__process.kinds for __process in base_processes]} to revert diff")
                                for index in batch_base_indices:
                                    target_data = self.org_data[target_columns].iloc[index : index + required_length]
                                    for base_process in base_processes:
                                        target_data = base_process(target_data)
                                    batch_base_values = pd.concat([batch_base_values, target_data.iloc[-1:]], axis=0)
                                batch_base_values = batch_base_values.values.reshape(1, *batch_base_values.shape)
                            else:
                                base_indices = [index - 1 for index in indices]
                                batch_base_values = self.org_data[target_columns].iloc[base_indices]
                        else:
                            base_indices = [index - 1 for index in indices]
                            batch_base_values = self.org_data[target_columns].iloc[base_indices]
                        r_data = process.revert(r_data, base_values=batch_base_values, columns=columns)
                    else:
                        raise Exception(f"Not implemented: {process.kinds}")
        return r_data


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

        super().__init__(df, columns, observation_length, device, processes, prediction_length, seed, is_training, randomize)

    def _output_func(self, batch_size):
        if type(batch_size) == int:
            index = self._indices[batch_size]
            ndx = self._output_indices(index)
            ans = self._data[self._columns].iloc[ndx].values.tolist()
            ans = torch.tensor(ans, device=self.device, dtype=torch.float)
            ans.reshape(self._prediction_length, 1, len(self._columns))
            time = self._data[self.time_column].iloc[ndx].values.tolist()
            time = torch.tensor(time, device=self.device, dtype=torch.int)
            time = time.reshape(self._prediction_length, 1, len(self.time_column))
            return ans, time
        elif type(batch_size) == slice:
            batch_indices = batch_size
            chunk_data = []
            time_chunk_data = []
            for index in self._indices[batch_indices]:
                ndx = self._output_indices(index)
                chunk_data.append(self._data[self._columns].iloc[ndx].values.tolist())
                time_chunk_data.append(self._data[self.time_column].iloc[ndx].values.tolist())

            return (
                torch.tensor(chunk_data, device=self.device, dtype=torch.float).transpose(0, 1),
                torch.tensor(time_chunk_data, device=self.device, dtype=torch.int).transpose(0, 1),
            )

    def _input_func(self, batch_size):
        if type(batch_size) == int:
            index = self._indices[batch_size]
            ndx = self._input_indices(index)
            src = self._data[ndx].values.tolist()
            src = torch.tensor(src, device=self.device, dtype=torch.float)
            src = src.reshape(self.observation_length, 1, len(self._columns))
            time = self._data[self.time_column].iloc[ndx].values.tolist()
            time = torch.tensor(time, device=self.device, dtype=torch.int)
            time = time.reshape(self._prediction_length, 1, len(self.time_column))
            return src, time
        elif type(batch_size) == slice:
            batch_indices = batch_size
            chunk_src = []
            time_chunk_data = []
            for index in self._indices[batch_indices]:
                ndx = self._input_indices(index)
                chunk_src.append(self._data[self._columns].iloc[ndx].values.tolist())
                time_chunk_data.append(self._data[self.time_column].iloc[ndx].values.tolist())

            return (
                torch.tensor(chunk_src, device=self.device, dtype=torch.float).transpose(0, 1),
                torch.tensor(time_chunk_data, device=self.device, dtype=torch.int).transpose(0, 1),
            )
