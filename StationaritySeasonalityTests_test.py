from Dataf import Dataf
from StationarityTests import StationarityTests
from TimeSeriesTools import TimeSeriesTools
from SeasonalityTest import SeasonalityTest

prices = Dataf.get_close_prices("AAPL", "2020-01-01", "2023-01-01")
log_returns = TimeSeriesTools.calculate_log_returns(prices)

StationarityTests.adf_test(log_returns)
StationarityTests.kpss_test(log_returns, regression='c')


SeasonalityTest.plot_decomposition(prices, freq=12) 
SeasonalityTest.plot_acf_for_seasonality(prices, lags=48)
