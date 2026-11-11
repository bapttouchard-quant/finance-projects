import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

data = yf.download("BTC-USD", start="2014-01-01", end="2026-11-10", auto_adjust=True)["Close"].squeeze()

genesis = pd.Timestamp(2009,1,3)

days_since_genesis = (data.index - genesis).days

log_days = np.log10(days_since_genesis)
log_price = np.log10(data)

X = sm.add_constant(log_days)
y = log_price
model = sm.OLS(y, X).fit()

today = pd.Timestamp(2026,11,10)
days_since_genesis_today = (today - genesis).days

log_days_today = np.log10(days_since_genesis_today)

fv_log = model.params.iloc[0] + model.params.iloc[1] * log_days_today
fv = 10 ** fv_log
actual_price = data.iloc[-1]

residual = model.resid
result = adfuller(residual)

residual.plot()
plt.show()
print(result)
print(fv, actual_price)
print(model.summary())
print(log_days[0], log_price.iloc[0])