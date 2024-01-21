import os
import random
import time
from collections.abc import Iterable
from multiprocessing import cpu_count, Array, Pipe, Process

import numpy as np
import pandas as pd
import torch
from pandas.tseries.frequencies import to_offset
from .simulator import DeterministicDealerModelV3


class AgentSimulationTrainDataGenerator:
    def __init__(
        self,
        agent_per_model: int,
        output_length: int,
        sample_timeindex,
        model_config=None,
        model_count: int = None,
        sampler_rule="MIN",
        batch_size=32,
        device=None,
        processes=None,
        seed=1017,
        is_training=True,
        dtype=torch.float,
        batch_first=False,
        open=True,
        high=True,
        low=True,
        close=True,
    ):
        if model_count is None:
            model_count = cpu_count()
        if isinstance(model_count, int) is False:
            raise TypeError("model_count should be int.")
        if model_count <= 0:
            raise ValueError("model_count should be greater than 0.")
        if agent_per_model <= 1:
            raise ValueError("total_seconds should be greater than 1.")
        if model_config is None:
            model_config = {}
        columns = []
        if open is True:
            columns.append("open")
        if high is True:
            columns.append("high")
        if low is True:
            columns.append("low")
        if close is True:
            columns.append("close")
        if len(columns) > 0:
            self.columns = columns
        else:
            raise ValueError("any of ohlc column is required.")
        try:
            self.max_delta = pd.to_timedelta(to_offset(sampler_rule))
            self.total_seconds = output_length * self.max_delta.total_seconds()
            self.output_length = output_length
        except Exception as e:
            raise ValueError(f"sampler_rule should be available as sampler rule: {e}")
        if device is None:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        pipes = []
        cpu_processes = []
        batch_sizes = []
        mini_batch = int(batch_size / model_count)
        for i in range(model_count):
            model = DeterministicDealerModelV3(agent_per_model, **model_config)
            parent_pipe, child_pipe = Pipe()
            process = Process(target=self.__child_process, args=(child_pipe, model))
            process.start()
            pipes.append(parent_pipe)
            cpu_processes.append(process)
            if i == model_count - 1:
                mini_batch = batch_size - mini_batch * i
            batch_sizes.append(mini_batch)
        self.__start_time = time.time()
        self.pipes = pipes
        self.cp = cpu_processes
        self.batch_sizes = batch_sizes
        self.max_mini_batch = max(self.batch_sizes)

        try:
            self.seed(seed)
            self.dtype = dtype
            self.batch_first = batch_first
            self.processes = processes
            self.is_training = is_training
            self.device = device
            self.data = None
            self.row_data = None
            self.sample_timemindex = sample_timeindex
            last_available_time = sample_timeindex[-1] - pd.Timedelta(seconds=self.total_seconds)
            available_timeindex = sample_timeindex[sample_timeindex < last_available_time]
            self.__max_index_to_use = len(available_timeindex)
            self.sampler_rule = sampler_rule
            # self.used_indices = []
        except Exception as e:
            self.terminate_subprocess()
            raise e

    def __child_process(self, pipe, model):
        command = 0
        thread = model.start()
        while command != -1:
            command = pipe.recv()
            if command == 1:
                pipe.send(model.price_history)
        model.end()
        del thread
        pipe.send(model.price_history)

    def terminate_subprocess(self):
        for pipe in self.pipes:
            pipe.send(-1)
        for p in self.cp:
            p.terminate()

    def run_process(self, data):
        if self.processes is not None:
            for process in self.processes:
                data = process(data)

    def __refresh_data(self):
        self.row_data = []
        for pipe in self.pipes:
            pipe.send(1)
            prices = pipe.recv()
            self.row_data.append(prices)

    def __acquire_data(self, data_index, required_length):
        self.__refresh_data()
        current_time = time.time()
        src = self.row_data[data_index]
        if len(src) < required_length:
            elapsed_seconds = current_time - self.__start_time
            length_per_sec = len(src) / elapsed_seconds
            lack_of_length = required_length - len(src)
            required_seconds = lack_of_length / length_per_sec + 1
            time.sleep(required_seconds)
            self.__acquire_data(data_index, required_length)

    def __iter__(self):
        self.__refresh_data()
        return self

    def get_next_indices(self):
        selected_index = random.randint(0, self.__max_index_to_use)
        time_indices = self.sample_timemindex[selected_index:]
        end_time = time_indices[0] + pd.Timedelta(seconds=self.total_seconds)
        next_indices = time_indices[time_indices <= end_time]
        # check if data has sufficient length.
        delta = end_time - next_indices[-1]
        if delta > self.max_delta:
            return self.get_next_indices()
        # check if indices includes holiday.
        time_diffs = (next_indices[1:] - next_indices[:-1]).total_seconds()
        org_length = len(time_diffs)
        out_condition = 3600
        time_diffs = time_diffs[time_diffs <= out_condition]
        # since simulation assume continuous trading, skip this index.
        if len(time_diffs) != org_length:
            return self.get_next_indices()
        # self.used_index.append(selected_value)
        return next_indices

    def __get_a_observation(self, data_index, indices):
        src = self.row_data[data_index]
        if len(src) >= len(indices):
            extra = len(src) - len(indices)
            index = random.randint(0, extra)
            data = src[index : index + len(indices)]
            srs = pd.Series(data, index=indices)
            df = srs.resample(self.sampler_rule).ohlc().ffill()
            return df[self.columns].values[:self.output_length]
        else:
            # wait until simulation output required length
            self.__acquire_data(data_index, len(indices))
            return self.__get_a_observation(data_index, indices)

    def __get_batch_observations(self):
        indices_array = []
        for _ in range(self.max_mini_batch):
            indices = self.get_next_indices()
            indices_array.append(indices)

        batch_src = []
        for data_index, mini_batch_size in enumerate(self.batch_sizes):
            for index in range(mini_batch_size):
                indices = indices_array[index]
                src = self.__get_a_observation(data_index, indices)
                batch_src.append(src)
        return batch_src

    def __next__(self):
        batch_data = self.__get_batch_observations()
        obs = torch.tensor(batch_data, dtype=self.dtype, device=self.device)
        if self.batch_first is True:
            return obs
        else:
            return obs.transpose(0, 1)

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

    def __del__(self):
        try:
            self.terminate_subprocess()
        except Exception:
            pass


class AgentSimulationWeeklyDataGenerator:
    def __init__(
        self,
        agent_per_model: int,
        timeindex,
        initial_price: float,
        model_configs: list = None,
        model_count: int = None,
        seed=1017,
    ):
        if model_count is None:
            model_count = cpu_count()
        if isinstance(model_count, int) is False:
            raise TypeError("model_count should be int.")
        if model_count <= 0:
            raise ValueError("model_count should be greater than 0.")
        if agent_per_model <= 1:
            raise ValueError("total_seconds should be greater than 1.")
        if isinstance(model_configs, dict) is True:
            model_configs = [model_configs for i in range(model_count)]
        elif isinstance(model_configs, Iterable) is True:
            if len(model_configs) != model_count:
                if len(model_configs) == 1:
                    model_configs = [model_configs[0] for i in range(model_count)]
                else:
                    raise ValueError("unexpected config value")
        else:
            raise TypeError("unexpected config type")
        self.model_configs = model_configs
        self.model_count = model_count
        self.agent_num = agent_per_model
        self.last_prices = [initial_price for _ in range(model_count)]
        self.__initialize_timeindex(timeindex)
        self.remaining_time_index = timeindex
        self.seed(seed)
        self.model_threads = [None for _ in range(model_count)]
        self.model_outputs = [None for _ in range(model_count)]
        self.__current_index = None

    def __initialize_timeindex(self, time_index):
        time_diffs = (time_index[1:] - time_index[:-1]).total_seconds()
        out_condition = 3600 * 30
        week_head_times = time_index[1:][time_diffs >= out_condition]
        self.week_head_times = week_head_times
        self.current_head_index = 0

    def __get_next_index(self, remaining_time_index):
        if len(self.week_head_times) > self.current_head_index:
            next_date = self.week_head_times[self.current_head_index + 1]
            next_index = remaining_time_index[remaining_time_index < next_date]
            remaining_time_index = remaining_time_index[remaining_time_index >= next_date]
            return next_index, remaining_time_index
        else:
            # last step. return remining indices
            return remaining_time_index, None

    def __create_model(self, model_index, required_length):
        if model_index >= self.model_count:
            raise IndexError("model index is out of range")
        config = self.model_configs[model_index]
        price = self.last_prices[model_index]
        config["initial_price"] = price
        shared_memory = Array(float, required_length)
        config["shared_memory"] = shared_memory

        model = DeterministicDealerModelV3(self.agent_num, **config)
        return model, shared_memory

    def __next__(self):
        next_index, next_remaining_index = self.__get_next_index(self.remaining_time_index)
        self.remaining_time_index = next_remaining_index
        for model_index in range(self.model_count):
            if len(self.model_threads) > model_index:
                thread = self.model_threads[model_index]
                if thread is not None:
                    thread.join()
                    tick_data = pd.Series(self.model_outputs[model_index], index=self.__current_index)
                    last_timestamp = self.__current_index[-1].timestamp()
                    file_name = f"tickdata_simulation_model{model_index}_{last_timestamp}.csv"
                    working_folder = os.getcwd()
                    file_path = os.join(working_folder, file_name)
                    tick_data.to_csv(file_path)
            model, shared_memory = self.__create_model(model_index, len(next_index))
            thread = model.start()
            self.model_threads[model_index] = thread
            self.model_outputs[model_index] = shared_memory
        self.__current_index = next_index

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
