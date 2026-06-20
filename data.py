# data.py - Download and prepare market data from Yahoo Finance
# Downloads historical data for major asset classes and calculates log returns

import time
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st


def _download_with_retry(ticker, period="5y", max_retries=3, base_delay=2):
    """
    Download data with retry + exponential backoff to survive Yahoo Finance
    rate limiting (common on cloud-hosted IPs like Streamlit Community Cloud).
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                return df
            last_err = ValueError(f"Empty data returned for {ticker}")
        except Exception as e:
            last_err = e

        # Exponential backoff before retrying: 2s, 4s, 8s...
        sleep_time = base_delay * (2 ** attempt)
        time.sleep(sleep_time)

    # All retries exhausted
    raise RuntimeError(
        f"Failed to download '{ticker}' after {max_retries} attempts. "
        f"Last error: {last_err}"
    )


@st.cache_data(ttl=3600, show_spinner="Fetching market data...")  # cache for 1 hour
def get_market_data():
    # Define tickers for key market indices and commodities
    tickers = {
        "SP500": "^GSPC",
        "OIL": "CL=F",
        "GOLD": "GC=F",
        "VIX": "^VIX"
    }

    data = {}
    failed = []

    # Download price history for each asset and calculate log returns
    for name, ticker in tickers.items():
        try:
            df = _download_with_retry(ticker, period="5y")
        except RuntimeError:
            failed.append(name)
            continue

        # Handle multi-level column index from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Calculate log returns for volatility and price dynamics analysis
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
        data[name] = df

        # Small delay between sequential calls to avoid tripping rate limits
        time.sleep(0.5)

    if failed:
        raise RuntimeError(
            f"Could not fetch market data for: {', '.join(failed)}. "
            "Yahoo Finance is likely rate-limiting this server. "
            "Please wait a few minutes and refresh."
        )

    return data
