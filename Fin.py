import yfinance as yf

nvda = yf.Ticker("NVDA")
# print(nvda.info)

data = nvda.history(period="5d", interval="5m")

print(data)