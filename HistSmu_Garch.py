

#First we load the necessary libraries
!pip install yfinance arch
import itertools
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from arch import arch_model
from scipy.stats import norm

#  We are pulling my financial data using yfinance via Yahoo Finance, we selected 3 stocks.
stocks = ['AAPL', 'MSFT', 'GOOGL']
data = yf.download(stocks, start='2020-01-01', end='2023-12-31')['Close']

print(data)

# For historical simulation, we obtain a ratio by dividing the difference between the previous price and the current price by the previous price.
data = data.dropna()
returns = []
for t in range(1, len(data)):
    r_t = (data.iloc[t] - data.iloc[t-1]) / data.iloc[t-1]
    returns.append(r_t)

print("Daily Returns (First 5 Rows):\n", returns[:5])

#We group them by month and look again.
returns_df = pd.DataFrame(returns, index=data.index[1:])
monthly_averages = returns_df.groupby(pd.Grouper(freq='ME')).mean()
print(monthly_averages)

##We group them by year and look again.

annual_averages = returns_df.groupby(pd.Grouper(freq='Y')).mean()
print(annual_averages)

#We calculate all-time averages

all_time_average = returns_df.mean()
print(all_time_average)

#We draw the graph of return rates based on monthly averages for 3 stocks.
plt.plot(monthly_averages.index, monthly_averages.values, label='monthly_averages')
plt.xlabel('Date')
plt.ylabel('Average Return')
plt.legend()
plt.show()

#We look at the change in the return rate on a single stock on a monthly, yearly and all-time basis.
stock_name = 'GOOGL'
one_returns = returns_df[stock_name]

monthly_averages_one = one_returns.groupby(pd.Grouper(freq='ME')).mean()
annual_averages_one = one_returns.groupby(pd.Grouper(freq='Y')).mean()
all_time_average_one = one_returns.mean()

plt.plot(monthly_averages_one.index, monthly_averages_one.values, label='monthly_averages')
plt.plot(annual_averages_one.index, annual_averages_one.values, label='annual_averages')
plt.axhline(y=all_time_average_one, color='r', linestyle='--', label='all_time_average')

plt.title(f'{stock_name}')
plt.xlabel('Date')
plt.ylabel('Average Return')
plt.legend()
plt.show

#We cross validate our Garch model to find the correct parameters, based on low AIC and BIC values.
series = returns_df['GOOGL'] * 100

results = []
for p, q in itertools.product(range(1, 4), range(1, 4)):
    try:
        m = arch_model(series, mean='Zero', vol='Garch', p=p, q=q, dist='normal')
        res = m.fit(disp='off')
        results.append({
            'p': p,
            'q': q,
            'AIC': res.aic,
            'BIC': res.bic,
            'LLF': res.loglikelihood
        })
    except:
        continue

df_results = pd.DataFrame(results)
best_aic = df_results.loc[df_results['AIC'].idxmin()]
best_bic = df_results.loc[df_results['BIC'].idxmin()]

print("The model with the lowest AIC:", best_aic.to_dict())
print("The model with the lowest BIC:", best_bic.to_dict())

#We construct the selected mode
V0 = data['GOOGL'][-1]
series = returns_df['GOOGL'] * 100
model = arch_model(series, mean='Zero', vol='Garch', p=1, q=2, dist='normal')
res = model.fit(disp='on')
print(res.summary())

#We make forecasts using the model
fcast = res.forecast(horizon=1)
sigma_next = np.sqrt(fcast.variance.values[-1,0]) / 100
z95 = norm.ppf(0.05)
var_garch = -V0 * sigma_next * z95

#We plot the graph of the fprecasting and the model
cond_vol = res.conditional_volatility / 100
plt.figure()
plt.plot(cond_vol)
plt.title('GARCH(1,2)')
plt.ylabel('volatility')
plt.xlabel('Date')
plt.show()