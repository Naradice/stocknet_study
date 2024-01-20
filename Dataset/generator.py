import random
import time
from multiprocessing import cpu_count, Pipe, Process

import numpy as np
import pandas as pd
import torch
from .simulator import DeterministicDealerModelV3


class AgentSimulationTrainDataGenerator:
    def __init__(
        self,
        agent_per_model: int,
        total_seconds: int,
        sample_timeindex,
        model_config=None,
        model_count: int = None,
        sampler_rule="MIN",
        device="cuda",
        processes=None,
        prediction_length=10,
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
        if total_seconds <= 0:
            raise ValueError("total_seconds should be greater than 0.")
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
        pipes = []
        cpu_processes = []
        for i in range(model_count):
            model = DeterministicDealerModelV3(agent_per_model, **model_config)
            parent_pipe, child_pipe = Pipe()
            process = Process(target=self.__child_process, args=(child_pipe, model))
            process.start()
            pipes.append(parent_pipe)
            cpu_processes.append(process)
        self.__start_time = time.time()
        self.pipes = pipes
        self.cp = cpu_processes

        try:
            self.seed(seed)
            self.total_seconds = total_seconds
            self.dtype = dtype
            self.batch_first = batch_first
            self.processes = processes
            self.is_training = is_training
            self.device = device
            self._prediction_length = prediction_length
            self.data = None
            self.row_data = None
            self.sample_timemindex = sample_timeindex
            last_available_time = sample_timeindex[-1] - pd.Timedelta(seconds=total_seconds)
            available_timeindex = sample_timeindex[sample_timeindex <= last_available_time]
            self.__max_index_to_use = len(available_timeindex)
            self.__indices = self.__get_next_indices()
            self.__data_index = 0
            self.__itr_index = 0
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

    def __acquire_data(self, required_length):
        self.__refresh_data()
        current_time = time.time()
        src = self.row_data[self.__data_index]
        if len(src) < required_length:
            elapsed_seconds = current_time - self.__start_time
            length_per_sec = len(src) / elapsed_seconds
            lack_of_length = required_length - len(src)
            required_seconds = lack_of_length / length_per_sec + 1
            time.sleep(required_seconds)
            self.__acquire_data(required_length)

    def __iter__(self):
        self.__refresh_data()
        return self

    def __get_next_indices(self):
        selected_index = random.randint(0, self.__max_index_to_use)
        time_indices = self.sample_timemindex[selected_index:]
        end_time = time_indices[0] + pd.Timedelta(seconds=self.total_seconds)
        next_indices = time_indices[time_indices <= end_time]
        # check if indices includes holiday.
        time_diffs = (next_indices[1:] - next_indices[:-1]).total_seconds()
        org_length = len(time_diffs)
        out_condition = 3600 * 23
        time_diffs = time_diffs[time_diffs <= out_condition]
        # since simulation assume continuous trading, skip this index
        if len(time_diffs) != org_length:
            return self.__get_next_indices()
        # self.used_index.append(selected_value)
        return next_indices

    def __get_data(self):
        self.__refresh_data()
        src = self.row_data[self.__data_index]
        if len(src) >= len(self.__indices):
            if self.__itr_index + len(self.__indices) > len(src):
                self.__data_index += 1
                self.__itr_index = 0
                if self.__data_index >= len(self.__row_data):
                    self.__data_index = 0
                    self.__indices = self.__get_next_indices()
                return self.__get_data()
            else:
                data = src[self.__itr_index : self.__itr_index + len(self.__indices)]
                srs = pd.Series(data, index=self.__indices)
                df = srs.resample(self.sampler_rule).ohlc().ffill()
                return df
        else:
            # wait until simulation output required length
            self.__acquire_data(len(self.__indices))
            return self.__get_data()

    def __next__(self):
        df = self.__get_data()
        return df
        # convert to torch,tensor

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
