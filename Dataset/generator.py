import argparse
import datetime
import os
import random
import time
from collections.abc import Iterable
from multiprocessing import cpu_count, Array, Pipe, Process

import numpy as np
import pandas as pd
import torch
from pandas.tseries.frequencies import to_offset

try:
    from simulator import DeterministicDealerModelV3
except ModuleNotFoundError:
    from .simulator import DeterministicDealerModelV3

try:
    import fprocess
except ImportError:
    from fprocess import fprocess


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
        if isinstance(model_config, Iterable) is True:
            if len(model_config) == model_count:
                model_configs = model_config
            else:
                raise ValueError("model_count should be the same as length of model configs.")
        elif isinstance(model_config, dict):
            model_configs = [model_config for _ in range(model_count)]
        else:
            raise TypeError(f"{type(model_config)} is not supported as model_config")
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
        required_length = 0
        if processes is not None:
            min_length = []
            for process in processes:
                min_length.append(process.get_minimum_required_length())
            required_length = max(min_length)
        else:
            processes = []
        self.processes = processes
        try:
            self.max_delta = pd.to_timedelta(to_offset(sampler_rule))
            self.total_seconds = (output_length + required_length) * self.max_delta.total_seconds()
            self.output_length = output_length
        except Exception as e:
            raise ValueError(f"sampler_rule should be available as sampler rule: {e}")
        if device is None:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        pipes = []
        cpu_processes = []
        batch_sizes = []
        self.target_batch_size = batch_size
        mini_batch = int(batch_size / model_count)
        for i in range(model_count):
            model_config = model_configs[i]
            model = DeterministicDealerModelV3(agent_per_model, **model_config)
            parent_pipe, child_pipe = Pipe()
            process = Process(target=self._child_process, args=(child_pipe, model))
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
        self.added_data = []
        self.additioan_batch_sizes = []

        try:
            self.seed(seed)
            self.dtype = dtype
            self.batch_first = batch_first
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

    @staticmethod
    def _child_process(pipe, model):
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
        try:
            self.__refresh_data()
        except Exception:
            raise StopIteration
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
            for process in self.processes:
                df = process(df)
            return df[self.columns].values[-self.output_length :].tolist()
        else:
            # wait until simulation output required length
            self.__acquire_data(data_index, len(indices))
            return self.__get_a_observation(data_index, indices)

    def __get_additional_observation(self, data_index):
        """get observation from separately added data. Typically it is added from saved simulation data."""
        src_df = self.added_data[data_index]
        index = random.randint(0, len(src_df) - self.output_length)
        return src_df.iloc[index : index + self.output_length].values.tolist()

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

        for data_index, mini_batch_size in enumerate(self.additioan_batch_sizes):
            for _ in range(mini_batch_size):
                src = self.__get_additional_observation(data_index)
                batch_src.append(src)
        return batch_src

    def __next__(self):
        try:
            batch_data = self.__get_batch_observations()
        except Exception:
            raise StopIteration
        obs = torch.tensor(batch_data, dtype=self.dtype, device=self.device)
        if self.batch_first is True:
            return obs
        else:
            return obs.transpose(0, 1)

    def _convert_tick_to_ohlc(self, tick_data: pd.Series):
        ohlc_df = tick_data.resample(self.sampler_rule).ohlc()
        ohlc_df = fprocess.convert.dropna_market_close(ohlc_df).ffill()
        for process in self.processes:
            ohlc_df = process(ohlc_df)
        ohlc_df.dropna(inplace=True)
        return ohlc_df

    def _update_batch_sizes(self):
        model_count = len(self.pipes)
        additional_data_count = len(self.added_data)

        # define batch sizes based on two params
        mini_batch = int(self.target_batch_size / (model_count + additional_data_count))
        self.batch_sizes = [mini_batch for _ in range(model_count)]
        self.max_mini_batch = mini_batch
        self.additioan_batch_sizes = []
        for i in range(additional_data_count):
            if i == additional_data_count - 1:
                mini_batch = self.target_batch_size - mini_batch * (model_count + i)
            self.additioan_batch_sizes.append(mini_batch)

    def add_data(self, tick_data: pd.Series):
        ohlc_df = self._convert_tick_to_ohlc(tick_data)
        # try if column has specified values
        self.added_data.append(ohlc_df[self.columns].copy())
        self._update_batch_sizes()

    def add_multiple_data(self, tick_data: list):
        for tick_srs in tick_data:
            ohlc_df = self._convert_tick_to_ohlc(tick_srs)
            # try if column has specified values
            self.added_data.append(ohlc_df[self.columns].copy())
        self._update_batch_sizes()

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

    def save_ticks(self, data_folder, data_update_required=True):
        if data_update_required:
            self.__refresh_data()
            print("data is updated.")
        date_str = datetime.datetime.now().isoformat(timespec="hours")
        date_str = date_str.replace(":", "-")
        os.makedirs(data_folder, exist_ok=True)
        for index, data in enumerate(self.row_data):
            sim_srs = pd.Series(data, index=self.sample_timemindex[: len(data)])
            sim_srs.to_csv(f"{data_folder}/tickdata_simulation_{date_str}_model_{index}.csv")

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
        if len(self.week_head_times) > (self.current_head_index - 1):
            next_date = self.week_head_times[self.current_head_index + 1]
            next_index = remaining_time_index[remaining_time_index < next_date]
            remaining_time_index = remaining_time_index[remaining_time_index >= next_date]
            self.current_head_index += 1
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
        shared_memory = Array("f", required_length)
        config["shared_memory"] = shared_memory

        model = DeterministicDealerModelV3(self.agent_num, **config)
        return model, shared_memory

    def __iter__(self):
        return self

    def __next__(self):
        next_index, next_remaining_index = self.__get_next_index(self.remaining_time_index)
        if next_remaining_index is None:
            raise StopIteration()
        self.remaining_time_index = next_remaining_index
        for model_index in range(self.model_count):
            if len(self.model_threads) > model_index:
                thread = self.model_threads[model_index]
                if thread is not None:
                    thread.join()
                    shared_memory = self.model_outputs[model_index]
                    tick_data = pd.Series(list(shared_memory.get_obj()), index=self.__current_index)
                    last_timestamp = self.__current_index[-1].timestamp()
                    file_name = f"tickdata_simulation_model{model_index}_{last_timestamp}.csv"
                    working_folder = os.getcwd()
                    file_path = os.path.join(working_folder, file_name)
                    tick_data.to_csv(file_path)
                    last_price = tick_data.iloc[-1]
                    self.last_prices[model_index] = float(last_price)
                    thread.close()
                    del shared_memory
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

    def terminate_subprocess(self):
        for thread in self.model_threads:
            if thread is not None:
                thread.terminate()

    def __del__(self):
        try:
            self.terminate_subprocess()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="cretes timeseries data simulated by multi agents.")
    parser.add_argument("target_file", type=str, help="filepath to read target tick data")
    parser.add_argument("-c", "--column", type=str, help="column to get initial price", default=None)
    parser.add_argument("-mn", "--model_num", type=int, help="number of parallel models", default=None)
    parser.add_argument("-an", "--agent_num", type=int, help="number of agent", default=300)
    parser.add_argument("-uv", "--upper_volatility", type=float, help="maximum value of volatility", default=0.02)
    parser.add_argument("-lv", "--lower_volatility", type=float, help="minimum value of volatility", default=0.01)
    parser.add_argument("-tu", "--trade_unit", type=float, help="a minimum unit to trade with (pips)", default=0.001)
    parser.add_argument("-s", "--spread", type=float, help="a spread for a simulation", default=1)
    parser.add_argument("-br", "--bull_ratio", metavar="0.0-1.0", help="represents how many users have bull position", default=0.5)
    parser.add_argument("-mtn", "--max_time_noise", type=float, help="specify max value of time noise", default=100)
    args = parser.parse_args()

    file = args.target_file
    column = args.column
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    timeindex = df.index
    if column is None:
        initial_price = 100.0
    elif column in df.columns:
        initial_price = df[column].iloc[0]
    else:
        raise ValueError(f"column {column} is not in file")
    del df

    model_num = args.model_num
    agent_num = args.agent_num
    max_volatility = args.upper_volatility
    min_volatility = args.lower_volatility
    trade_unit = args.trade_unit
    spread = args.spread
    bull_ratio = args.bull_ratio
    bull_ratio = np.clip(bull_ratio, 0.0, 1.0)
    num_of_bull = round(agent_num * bull_ratio)
    num_of_bear = agent_num - num_of_bull
    agent_positions = [1 for i in range(num_of_bull)]
    bear_agent_positions = [-1 for i in range(num_of_bear)]
    agent_positions.extend(bear_agent_positions)

    config = {
        "max_volatility": max_volatility,
        "min_volatility": min_volatility,
        "trade_unit": trade_unit,
        "spread": spread,
        "initial_positions": agent_positions,
    }

    generator = AgentSimulationWeeklyDataGenerator(
        agent_per_model=agent_num, model_count=model_num, timeindex=timeindex, initial_price=initial_price, model_configs=config
    )

    for _ in generator:
        pass


if __name__ == "__main__":
    main()
