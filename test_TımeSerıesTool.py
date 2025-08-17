from Dataf import Dataf
from TimeSeriesTools import TimeSeriesTools

prices = Dataf.get_close_prices("AAPL", "2020-01-01", "2023-01-01")
log_returns = TimeSeriesTools.calculate_log_returns(prices)
sma_50 = TimeSeriesTools.calculate_sma(prices, window=50)
print(log_returns.head())
print(sma_50.head(60))
