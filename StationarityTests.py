import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

class StationarityTests:
    
  
    def adf_test(series: pd.Series, verbose: bool = True) -> dict:
       
        result = adfuller(series.dropna())
        output = {
            'ADF Statistic': result[0],
            'p-value': result[1],
            'lags_used': result[2],
            'n_obs': result[3],
            'critical_values': result[4],
            'stationary': result[1] < 0.05
        }
        if verbose:
            print("\n### ADF Test Results ###")
            for key, value in output.items():
                print(f"{key}: {value}")
        return output

    def kpss_test(series: pd.Series, regression: str = 'c', verbose: bool = True) -> dict:
        
        result = kpss(series.dropna(), regression=regression, nlags="auto")
        output = {
            'KPSS Statistic': result[0],
            'p-value': result[1],
            'lags_used': result[2],
            'critical_values': result[3],
            'stationary': result[1] > 0.05
        }
        if verbose:
            print("\n### KPSS Test Results ###")
            for key, value in output.items():
                print(f"{key}: {value}")
        return output
