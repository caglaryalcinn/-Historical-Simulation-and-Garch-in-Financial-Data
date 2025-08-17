import pandas as pd
import numpy as np

class TimeSeriesTools:
    
    def calculate_log_returns(prices: pd.Series, dropna: bool = True) -> pd.Series:
      
        log_ret = np.log(prices / prices.shift(1))
        return log_ret.dropna() if dropna else log_ret

    
    def calculate_sma(prices: pd.Series, window: int = 20, dropna: bool = False) -> pd.Series:
       
        sma = prices.rolling(window=window).mean()
        return sma.dropna() if dropna else sma

    
    def calculate_ema(prices: pd.Series, span: int = 20, dropna: bool = False) -> pd.Series:
      
        ema = prices.ewm(span=span, adjust=False).mean()
        return ema.dropna() if dropna else ema

  
    def difference_series(prices: pd.Series, lag: int = 1, dropna: bool = True) -> pd.Series:
       
        diff = prices.diff(periods=lag)
        return diff.dropna() if dropna else diff
