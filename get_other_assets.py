import yfinance as yf
import pandas as pd
import time
assets = ["ETH-USD", "XRP-USD", "DOGE-USD", "SOL-USD", "ADA-USD", "BCH-USD", "LTC-USD","EURUSD=X", "JPY=X", "AAPL", "NVDA","TSLA",]
for asset in (assets):
    df = yf.download(asset, start="2024-01-01", end="2024-12-31", interval="1h")
    data = df.copy()
    data.columns = data.columns.get_level_values('Price')
    data.columns = ["close", "open", "high", "low", "volume"]
    data['timestamp'] = pd.to_datetime(data.index).astype(int) // 10**9
    data = data.reset_index(drop=True)

    data.to_csv(f"dataset/other_assets/{asset.replace('=', '-')}.csv")

    time.sleep(10)