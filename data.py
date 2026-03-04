# data.py - Download and prepare market data from Yahoo Finance
# Downloads historical data for major asset classes and calculates log returns

import yfinance as yf
import pandas as pd
import numpy as np

def get_market_data():
    # Define tickers for key market indices and commodities
    tickers = {
        "SP500": "^GSPC",
        "OIL": "CL=F",
        "GOLD": "GC=F",
        "VIX": "^VIX"
    }

    data = {}

    # Download price history for each asset and calculate log returns
    for name, ticker in tickers.items():
        df = yf.download(ticker, period="5y")

        # Handle multi-level column index from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Calculate log returns for volatility and price dynamics analysis
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
        data[name] = df

    return data

