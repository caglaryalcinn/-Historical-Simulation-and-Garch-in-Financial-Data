import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, nan 

class HistoricalSimulation:


    def __init__(self):
        pass

    def _filter_data(data, start=None, end=None):
        """
        Filters a DataFrame/Series between start and end dates.
        data sholud be pd.Series or pd.DataFrame: Time series data to filter.
        start sholud be str or pd.Timestamp, optional: Start date for filtering.
        end sholud be str or pd.Timestamp, optional : End date for filtering.
         """
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise TypeError("Input data must be a Pandas Series or DataFrame.")

        if isinstance(data, pd.DataFrame) and data.shape[1] == 1:
            data = data.iloc[:, 0]
        elif isinstance(data, pd.DataFrame) and data.shape[1] > 1:
            raise ValueError("This class is currently designed to work with a single time series and Pandas Series.")

        data = data.dropna()
        if start is not None and end is not None:
            return data.loc[start:end]
        elif start is not None:
            return data.loc[start:]
        elif end is not None:
            return data.loc[:end]
        else:
            return data

    def _calculate_returns(prices, method='simple'):
       
        if prices.empty or len(prices) < 2:
            return pd.Series(dtype=float) 

        if method == 'simple':
            returns = prices.pct_change().dropna()
        elif method == 'log':
            returns = np.log(prices / prices.shift(1)).dropna()
        else:
            raise ValueError("Method must be 'simple' or 'log'.")
        return returns

    def simulate(self, y, start=None, end=None, initial_value=1_000_000, method='simple'):
        """
        Simulates portfolio value over time using historical returns.
         y (pd.Series): Price time series.
        start (str or pd.Timestamp, optional): Simulation start date.
        end (str or pd.Timestamp, optional): Simulation end date.
        initial_value (float): Starting value of the portfolio.
        method (str): Return calculation method ('simple' or 'log').
        """
        filtered_prices = HistoricalSimulation._filter_data(y, start, end)
        returns = HistoricalSimulation._calculate_returns(filtered_prices, method=method)

        if returns.empty:
             print(f"Warning (simulate {method}): Cannot calculate returns or insufficient data after filtering.")
             return pd.Series(dtype=float, name=f"Portfolio Value ({method.capitalize()})")

        if method == 'simple':
            value = initial_value * (1 + returns).cumprod()
        elif method == 'log':
            value = initial_value * np.exp(returns.cumsum())

        first_valid_index = returns.index[0]
        freq = pd.infer_freq(filtered_prices.index)
        day_before = None
        if freq == 'B': day_before = first_valid_index - pd.tseries.offsets.BusinessDay()
        elif freq:
             try: offset = pd.tseries.frequencies.to_offset(freq); day_before = first_valid_index - offset
             except ValueError: day_before = first_valid_index - pd.Timedelta(days=1)
        else: day_before = first_valid_index - pd.Timedelta(days=1)

        initial_point = None
        if value.index.tz is not None and day_before.tz is None:
             initial_point = pd.Series([initial_value], index=[day_before.tz_localize(value.index.tz)])
        elif value.index.tz is None and day_before.tz is not None:
             initial_point = pd.Series([initial_value], index=[day_before.tz_convert(None)])
        elif day_before is not None:
             initial_point = pd.Series([initial_value], index=[day_before])

        if initial_point is not None:
            portfolio_value = pd.concat([initial_point, value])
            # Remove potential duplicate index from concat
            portfolio_value = portfolio_value[~portfolio_value.index.duplicated(keep='last')]
        else: # its for safety
            portfolio_value = value

        portfolio_value.name = f"Portfolio Value ({method.capitalize()})"
        return portfolio_value

    def calculate_volatility(self, y, start=None, end=None, method='ewma', ewma_lambda=0.94, window=None, return_series=False):
        """
        Calculates volatility using either EWMA or simple rolling standard deviation.
        """
        filtered_prices = HistoricalSimulation._filter_data(y, start, end)
        returns = HistoricalSimulation._calculate_returns(filtered_prices, method='simple') # Volatility typically uses simple returns

        if returns.empty:
             print(f"Warning (calculate_volatility {method}): Cannot calculate returns for volatility.")
             return pd.Series(dtype=float) if return_series else nan

        volatility = None
        vol_name = "Volatility"

        if method == 'ewma':
            if not (0 < ewma_lambda < 1):
                raise ValueError("ewma_lambda must be between 0 and 1.")
            squared_returns = returns ** 2
            ####### alpha = 1 - lambda
            ewma_var = squared_returns.ewm(alpha=1 - ewma_lambda, adjust=True).mean()
            volatility = np.sqrt(ewma_var)
            vol_name = f"EWMA Volatility (λ={ewma_lambda})"
        elif method == 'simple':
            if window is not None and window <= 0:
                 raise ValueError("Window must be a positive integer for simple volatility.")
            ## Calculate rolling or expanding standard deviation of returns
            volatility = returns.rolling(window=window).std() if window else returns.expanding().std()
            ### Often annualized: multiply by sqrt of trading_days_per_year(252)
            #### volatility = sqrt(252)  
            vol_name = f"Rolling Volatility (window={window})" if window else "Expanding Volatility"
        else:
            raise ValueError("Volatility method must be 'ewma' or 'simple'.")

        volatility.name = vol_name

        if return_series:
            return volatility
        else:
            ### Return the last valid volatility value
            return volatility.iloc[-1] if not volatility.empty else nan


    def calculate_var(self, y, start=None, end=None, confidence_level=0.99, current_value=None, return_method='simple'):
        """
        Calculates Value at Risk (VaR) using the Historical Simulation method.
        """
        historical_prices = HistoricalSimulation._filter_data(y, start, end)

        if historical_prices.empty or len(historical_prices) < 2:
            print(f"Warning (VaR): Insufficient historical data (< 2) for VaR calculation. {len(historical_prices)} points remain.")
            return nan

        if current_value is None:
            current_value = historical_prices.iloc[-1]
            print(f"Warning (VaR): current_value not specified, using last price ({current_value:.2f}).")
        elif current_value <= 0:
            print("Warning (VaR): current_value must be positive.")
            return nan

        returns = HistoricalSimulation._calculate_returns(historical_prices, method=return_method)

        if returns.empty:
            print(f"Warning (VaR): Could not calculate historical returns using method '{return_method}'.")
            return nan

        simulated_changes = current_value * returns

        ################### Find the loss percentile 1% for 99% confidence
        loss_percentile = 1.0 - confidence_level

        var_value = simulated_changes.quantile(loss_percentile)

        return abs(var_value) if not pd.isna(var_value) else nan


    def calculate_es(self, y, start=None, end=None, confidence_level=0.99, current_value=None, return_method='simple'):
        """
        Calculates Expected Shortfall (ES) / Conditional Value at Risk (CVaR)
        using the Historical Simulation method. ES is the expected average loss
        when VaR is exceeded.
        """
        historical_prices = HistoricalSimulation._filter_data(y, start, end)

        if historical_prices.empty or len(historical_prices) < 2:
            print(f"Warning (ES): Insufficient historical data (< 2) for ES calculation. {len(historical_prices)} points remain.")
            return nan

        if current_value is None:
            current_value = historical_prices.iloc[-1]
            print(f"Warning (ES): current_value not specified, using last price ({current_value:.2f}).")
        elif current_value <= 0:
            print("Warning (ES): current_value must be positive.")
            return nan

        returns = HistoricalSimulation._calculate_returns(historical_prices, method=return_method)

        if returns.empty:
            print(f"Warning (ES): Could not calculate historical returns using method '{return_method}'.")
            return nan

        simulated_changes = current_value * returns

        loss_percentile = 1.0 - confidence_level
        var_value = simulated_changes.quantile(loss_percentile)

        if pd.isna(var_value):
            print("Warning (ES): Cannot calculate ES because VaR calculation failed.")
            return nan

        tail_losses = simulated_changes[simulated_changes <= var_value]

        es_value = tail_losses.mean()

        return abs(es_value) if not pd.isna(es_value) else nan


    def plot_portfolios(self, **portfolios):
        if not portfolios:
            print("No portfolios provided for plotting.")
            return
        plt.figure(figsize=(12, 6))
        for name, series in portfolios.items():
            if isinstance(series, pd.Series) and not series.empty:
                plt.plot(series.index, series.values, label=name)
            else:
                print(f"Warning: '{name}' is not a valid or non-empty Series, skipping plot.")
        plt.title("Portfolio Simulations")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


