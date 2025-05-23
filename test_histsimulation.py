from HistoricalSimulation import HistoricalSimulation
from Dataf import Dataf

# get data
close_prices = Dataf.get_close_prices('AAPL', start='2023-01-01', end='2024-01-01')

# basic simulation 
portfolio_all = HistoricalSimulation.simulate_simple(close_prices)
print("All Data Simulation (last 5 values):")
print(portfolio_all.tail())


# log simulation with parameter filter
portfolio_log = HistoricalSimulation.simulate_log(close_prices, start='2023-07-01')
print("LogLastHalf Simulation (last 5 values):")
print(portfolio_log.tail())



# draw
HistoricalSimulation.plot_portfolios(
    basic_all=portfolio_all,
    log_half=portfolio_log
)
# get ewmas
son_gun_ewma = HistoricalSimulation.calculate_ewma_volatility(close_prices)
print(f"Last Day EWMA Volatility: {son_gun_ewma:.6f}")

ewma_series = HistoricalSimulation.calculate_ewma_volatility(close_prices, return_series=True)
print(ewma_series.head())
