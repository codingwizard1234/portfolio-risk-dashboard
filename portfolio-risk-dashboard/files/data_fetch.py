import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_data(tickers, start="2020-01-01", end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    
    # auto_adjust=True by default → 'Close' is already adjusted
    data = yf.download(tickers, start=start, end=end, progress=False)['Close']
    return data

if __name__ == "__main__":
    tickers = ["TSLA","RELIANCE.NS"]
    df = fetch_data(tickers, "2024-07-25", "2025-07-27")

    print("Shape:", df.shape)
    print(df.head())
    
    
    df.to_csv("stock_data.csv")
    print("✅ Data saved to stock_data.csv")
