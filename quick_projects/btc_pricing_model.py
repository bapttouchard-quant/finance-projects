import yfinance as yf

data = yf.download("BTC-USD", start="2014-01-01", end="2026-06-01", auto_adjust=True)["Close"]

print(data.describe())