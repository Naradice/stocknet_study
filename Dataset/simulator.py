import argparse
import datetime
import os
import random
from collections.abc import Iterable

import numpy as np
import pandas as pd


class DeterministicDealerModelV1:
    def __init__(
        self,
        num_agent,
        max_volatility=0.02,
        min_volatility=0.01,
        trade_unit=0.001,
        initial_price=100,
        spread=1,
        initial_positions=None,
        tick_time=0.001,
        experimental=False,
        time_noise_method=None,
        max_noise_factor=1,
    ) -> None:
        # min_vol should be greater than 0
        # agent tendency: agent change his order prices based on the tendency
        tendency = pd.Series([random.uniform(min_volatility, max_volatility) for i in range(num_agent)], dtype=float)
        if trade_unit < 1:
            do_round = True
            decimal = 0.1
            decimal_num = 1
            while True:
                check = trade_unit / decimal
                if check >= 1 - trade_unit:
                    break
                decimal *= 0.1
                decimal_num += 1
                if decimal_num > 100:
                    do_round = False
                    break
            if do_round is True:
                tendency = tendency.round(decimal_num)
        prices = pd.Series([random.uniform(initial_price, initial_price + spread) for i in range(num_agent)], dtype=float)
        if initial_positions is None:
            position_trends = pd.Series([random.choice([-1, 1]) for i in range(num_agent)], dtype=int)
        elif len(initial_positions) == num_agent:
            position_trends = pd.Series(initial_positions, dtype=int)
        else:
            raise ValueError("initial position is invalid.")
        self.agent_df = pd.concat([tendency, position_trends, prices], axis=1, keys=["tend", "position", "price"], names=["id"])
        self.spread = spread
        self.market_price = initial_price + self.spread
        self.__initial_price = initial_price
        self.__min_vol = min_volatility
        self.__max_vol = max_volatility
        self.trade_unit = trade_unit
        self.tick_time = 0.0
        self.tick_time_unit = tick_time
        self.max_noise_factor = max_noise_factor
        if experimental is True:
            self.simulate = self.__freq_advance_simulate
        else:
            self.simulate = self.__ref_simulate

        if time_noise_method is None:
            self.__get_time_noise = lambda: 1
        elif time_noise_method == "uniform":
            self.__get_time_noise = self.__get_uniform_noise
        elif time_noise_method == "exp":
            self.__noise_factors = list(range(1, self.max_noise_factor + 1))
            self.__noise_weights = [1 / (i + 1) for i in range(self.max_noise_factor)]
            self.__get_time_noise = self.__get_weighted_noise
        elif callable(time_noise_method) is True:
            self.__get_time_noise = time_noise_method
        else:
            raise ValueError("this method is not defined")

    def __get_uniform_noise(self):
        random_number = random.uniform(1, self.max_noise_factor)
        return random_number

    def __get_weighted_noise(self):
        random_number = random.choices(self.__noise_factors, weights=self.__noise_weights)[0]
        return random_number

    def advance_order_price(self):
        self.agent_df.price += self.agent_df.position * self.agent_df.tend
        # print("*kept*")
        # print(self.agent_df)
        return self.agent_df.price

    def contruct(self):
        ask_agents = self.agent_df.loc[self.agent_df.position == 1]
        bought_agent_id = ask_agents.price.idxmax()
        ask_order_price = ask_agents.price[bought_agent_id]

        bid_agents = self.agent_df.loc[self.agent_df.position == -1]
        sold_agent_id = bid_agents.price.idxmin()
        bid_order_price = bid_agents.price[sold_agent_id] + self.spread

        if ask_order_price >= bid_order_price:
            # print("---contructed!!---")
            self.market_price = (((ask_order_price + bid_order_price) / 2) // self.trade_unit) * self.trade_unit
            # print(bought_agent_id, sold_agent_id)
            self.agent_df.loc[bought_agent_id, "position"] = -1
            self.agent_df.loc[sold_agent_id, "position"] = 1
            # print(self.agent_df)
            return self.market_price
        return None

    def __ref_simulate(self, total_seconds=100, length=None):
        self.price_history = [self.market_price]
        tick_times = [self.tick_time]
        if isinstance(length, int):
            while len(self.tick_times) < length:
                self.tick_time += self.tick_time_unit
                price = self.contruct()
                if price is not None:
                    self.price_history.append(price)
                    tick_times.append(self.tick_time)
                else:
                    self.advance_order_price()
        else:
            while self.tick_time < total_seconds:
                random_span_factor = self.__get_time_noise()
                self.tick_time += self.tick_time_unit * random_span_factor
                price = self.contruct()
                if price is not None:
                    self.price_history.append(price)
                    tick_times.append(self.tick_time)
                else:
                    self.advance_order_price()
        price_hist_df = pd.DataFrame(self.price_history, columns=["price"])
        tick_times = np.asarray(tick_times)
        self.tick_time = 0
        return price_hist_df, tick_times

    def __freq_advance_simulate(self, total_seconds=100, length=None):
        self.price_history = [self.market_price]
        tick_times = [self.tick_time]
        if isinstance(length, int):
            while len(self.tick_times) < length:
                self.tick_time += self.tick_time_unit * random_span_factor
                self.advance_order_price()
                price = self.contruct()
                if price is not None:
                    self.price_history.append(price)
                    tick_times.append(self.tick_time)
        else:
            while self.tick_time < total_seconds:
                random_span_factor = self.__get_time_noise()
                self.tick_time += self.tick_time_unit * random_span_factor
                self.advance_order_price()
                price = self.contruct()
                if price is not None:
                    self.price_history.append(price)
                    tick_times.append(self.tick_time)
        price_hist_df = pd.DataFrame(self.price_history, columns=["price"])
        tick_times = np.asarray(tick_times)
        self.tick_time = 0
        return price_hist_df, tick_times


class DeterministicDealerModelV3(DeterministicDealerModelV1):
    def __init__(
        self,
        num_agent,
        max_volatility=0.02,
        min_volatility=0.01,
        trade_unit=0.001,
        initial_price=100,
        spread=1,
        initial_positions=None,
        tick_time=0.001,
        experimental=False,
        time_noise_method=None,
        max_noise_factor=1,
        dealer_sensitive=None,
        wma=1,
        dealer_sensitive_min=-3.5,
        dealer_sensitive_max=-1.5,
    ) -> None:
        """_summary_

        Args:
            dealer_sensitive (int|Iterable, optional): this values represent sensitivity of each agent for past prices. greater/less than 0 means follower/contrarian. Defaults to None and initialized with random values by normal dist with mean and sigma
            wma (int|Iterable, optional): this values represent how long agents will be affected by past values. Defaults to 1.
        """
        super().__init__(
            num_agent,
            max_volatility,
            min_volatility,
            trade_unit,
            initial_price,
            spread,
            initial_positions,
            tick_time,
            experimental=experimental,
            time_noise_method=time_noise_method,
            max_noise_factor=max_noise_factor,
        )
        self.price_history = [self.market_price]
        self.tick_times = [self.tick_time]

        if dealer_sensitive is None:
            # create array with random values between -1 to 1
            self.dealer_sensitive = np.asarray([random.uniform(dealer_sensitive_min, dealer_sensitive_max) for i in range(num_agent)])
        elif isinstance(dealer_sensitive, (int, float)):
            # create array with fixed value
            self.dealer_sensitive = np.asarray([dealer_sensitive for i in range(num_agent)])
        elif isinstance(dealer_sensitive, Iterable):
            # use provided array
            self.dealer_sensitive = np.asanyarray(dealer_sensitive)
        else:
            raise TypeError("dealer_sensitive must be either float or Iterable object")

        if isinstance(wma, int):
            # create weight array with random values between 0 to 1
            self.weight_array = np.asarray([random.uniform(0, 1) for i in range(wma)])
            self.__total_weight = np.sum(self.weight_array)
            self.wma = wma
        elif isinstance(wma, Iterable):
            #  use wma as weight array
            self.weight_array = np.asarray(wma)
            self.__total_weight = np.sum(self.weight_array)
            self.wma = len(wma)
        else:
            raise TypeError("wma must be either int or Iterable object")

    def __wma(self):
        if len(self.price_history) >= self.wma + 1:
            wma_price = 0.0
            for i in range(1, self.wma + 1):
                price_diff = self.price_history[-i] - self.price_history[-i - 1]
                wma_price += self.weight_array[i - 1] * price_diff
            return wma_price / self.__total_weight
        else:
            return 0

    def advance_order_price(self):
        follow_factors = self.dealer_sensitive * self.__wma()
        self.agent_df.price += self.agent_df.position * self.agent_df.tend + follow_factors
        return self.agent_df.price


def valid_percentage(value):
    fvalue = float(value)
    if 0 <= fvalue <= 1:
        return fvalue
    else:
        raise argparse.ArgumentTypeError(f"expected 0 to 1. {value} is provided.")


def main():
    output_file_name = f"multiagent_result_{datetime.datetime.timestamp(datetime.datetime.now())}.csv"

    parser = argparse.ArgumentParser(description="cretes timeseries data simulated by multi agents.")
    parser.add_argument("simulation_num", type=int, help="specify how many times run simulations. data length is depends on end_condition value")
    parser.add_argument(
        "-ec",
        "--end_condition",
        type=str,
        choices=["seconds", "length"],
        help="if seconds, simulation end when 0.001 * times over the simulation_num. if length, simulation end when data has the simulation_num length.",
        default="seconds",
    )
    parser.add_argument("-o", "--out", type=str, help="path to output a result.", default="./")
    parser.add_argument("-an", "--agent_num", type=int, help="number of agent", default=300)
    parser.add_argument("-uv", "--upper_volatility", type=float, help="maximum value of volatility", default=0.02)
    parser.add_argument("-lv", "--lower_volatility", type=float, help="minimum value of volatility", default=0.01)
    parser.add_argument("-tu", "--trade_unit", type=float, help="a minimum unit to trade with (pips)", default=0.001)
    parser.add_argument("-ip", "--initial_price", type=float, help="a initial price (not a pips)", default=100)
    parser.add_argument("-s", "--spread", type=float, help="a spread for a simulation", default=1)
    parser.add_argument(
        "-br", "--bull_ratio", type=valid_percentage, metavar="0.0-1.0", help="represents how many users have bull position", default=0.5
    )
    parser.add_argument(
        "-tnm",
        "--time_noise_method",
        type=str,
        choices=["none", "uniform", "weighted"],
        help="method to add a noise to timeindex randomly",
        default="weighted",
    )
    parser.add_argument("-mtn", "--max_time_noise", type=float, help="specify max value of time noise", default=100)
    args = parser.parse_args()

    simulation_num = int(args.simulation_num)
    if args.end_condition == "seconds":
        options = {"total_seconds": simulation_num}
    else:
        options = {"length": simulation_num}

    if os.path.isabs(args.out):
        output_folder_path = args.out
    else:
        working_folder = os.getcwd()
        output_folder_path = os.path.abspath(os.path.join(working_folder, args.out))
    os.makedirs(output_folder_path, exist_ok=True)
    output_file_path = os.path.join(output_folder_path, output_file_name)

    agent_num = args.agent_num
    max_volatility = args.upper_volatility
    min_volatility = args.lower_volatility
    trade_unit = args.trade_unit
    initial_price = args.initial_price
    spread = args.spread
    bull_ratio = args.bull_ratio
    bull_ratio = np.clip(bull_ratio, 0.0, 1.0)
    num_of_bull = round(agent_num * bull_ratio)
    num_of_bear = agent_num - num_of_bull
    agent_positions = [1 for i in range(num_of_bull)]
    bear_agent_positions = [-1 for i in range(num_of_bear)]
    agent_positions.extend(bear_agent_positions)
    time_noise_method = args.time_noise_method
    if time_noise_method == "none":
        time_noise_method = None
    elif time_noise_method == "weighted":
        time_noise_method = "exp"
    max_noise_factor = args.max_time_noise

    model = DeterministicDealerModelV3(
        num_agent=agent_num,
        max_volatility=max_volatility,
        min_volatility=min_volatility,
        trade_unit=trade_unit,
        initial_price=initial_price,
        spread=spread,
        initial_positions=agent_positions,
        time_noise_method=time_noise_method,
        max_noise_factor=max_noise_factor,
    )

    prices, ticks = model.simulate(**options)
    prices.index = ticks

    prices.to_csv(output_file_path, index=True)


if __name__ == "__main__":
    main()
