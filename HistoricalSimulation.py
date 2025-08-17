import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class HistoricalSimulation:

    def _filter_data(close_prices, start=None, end=None):
        if start is not None and end is not None:
            return close_prices.loc[start:end]
        elif start is not None:
            return close_prices.loc[start:]
        elif end is not None:
            return close_prices.loc[:end]
        else:
            return close_prices

    def simulate_simple(close_prices, start=None, end=None, initial_value=1_000_000):
        close_prices = HistoricalSimulation._filter_data(close_prices, start, end)
        returns = close_prices.pct_change().dropna()
        value = initial_value * (1 + returns).cumprod()
        return value

    def simulate_log(close_prices, start=None, end=None, initial_value=1_000_000):
        close_prices = HistoricalSimulation._filter_data(close_prices, start, end)
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        value = initial_value * np.exp(log_returns.cumsum())
        return value


    def calculate_ewma_volatility(close_prices, start=None, end=None, ewma_lambda=0.94, return_series=False):
        close_prices = HistoricalSimulation._filter_data(close_prices, start, end)
        returns = close_prices.pct_change().dropna()
        squared_returns = returns ** 2
        ewma_var = squared_returns.ewm(alpha=1 - ewma_lambda).mean()
        ewma_vol = np.sqrt(ewma_var)

        if isinstance(ewma_vol, pd.DataFrame):
            if ewma_vol.shape[1] == 1:
                ewma_vol = ewma_vol.iloc[:, 0]
            else:
                raise ValueError("please enter one stock")

        if return_series:
            return ewma_vol 
        else:
            return ewma_vol.iloc[-1]  


    def plot_portfolios(**portfolios):
        plt.figure(figsize=(12, 6))
        for name, series in portfolios.items():
            plt.plot(series, label=name)
        plt.title("Simulations")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        plt.show()


