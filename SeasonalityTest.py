import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

class SeasonalityTest:

    def plot_decomposition(series: pd.Series, freq: int = 12):
      
        decomposition = seasonal_decompose(series.dropna(), model='additive', period=freq)
        decomposition.plot()
        plt.suptitle("STL Decomposition", fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_acf_for_seasonality(series: pd.Series, lags: int = 48):
      
        plt.figure(figsize=(10, 5))
        plot_acf(series.dropna(), lags=lags)
        plt.title("Autocorrelation Plot")
        plt.tight_layout()
        plt.show()
