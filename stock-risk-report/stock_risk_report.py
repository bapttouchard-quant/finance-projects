import yfinance as yf
import pandas as pd
import numpy as np

tickers = [input("Stock to analyse: ").strip().upper(), 'SPY']
data = yf.download(tickers, start='2000-01-01', end='2026-01-01', auto_adjust=True)["Close"]

returns = data.pct_change().dropna()

if returns.empty:
    raise SystemExit("No data available, try again later...")

daily_vol = returns.std()
volatility = returns.std() * np.sqrt(252)

#One-day VaR calculations / 99% Confidence
p, z = 10000, 2.33
var99 = p * z * daily_vol

#Historical VaR
hist_var = -(returns.quantile(0.01) * p)

threshold = -(var99 / p)
breach_rate = ((returns < threshold).mean())*100

#Sharpe Ratio Calculation
irx = (yf.download("^IRX", start='2000-01-01', end='2026-01-01', auto_adjust=True)["Close"].squeeze().reindex(returns.index).ffill().shift(1))/252

excess = returns.sub(irx/100, axis=0)
sharpe = excess.mean() / excess.std() * np.sqrt(252)

#Annualized Return
years = len(returns) / 252
cagr = ((1 + returns).prod())**(1/years) - 1

#Drawdown Calculation
price = (1 + returns).cumprod()
peak = price.cummax()
below_peak = (price / peak) - 1
max_dd = below_peak.min()

report = pd.DataFrame({
    "1-Day VaR 99% (Parametric)": var99.map("${:,.0f}".format),
    "1-Day VaR 99% (Historical)": hist_var.map("${:,.0f}".format),
    "VaR Breach Rate (Expected 1%)": breach_rate.map("{:.2f}%".format),
    "Sharpe Ratio": sharpe.map("{:.2f}".format),
    "Annualized Return (CAGR)": cagr.map("{:.1%}".format),
    "Max Drawdown": max_dd.map("{:.1%}".format),
    "Max Drawdown Date": below_peak.idxmin().dt.date,
})

print(report.T)



