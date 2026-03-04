# features.py - Feature engineering for risk prediction model
# Combines multiple asset returns and technical indicators to create ML features

import numpy as np

def create_features(market_data):
    # Start with S&P 500 price data as base series
    df = market_data["SP500"].copy()

    # Add cross-asset returns to capture geopolitical and commodity risks
    df["Oil_Return"] = market_data["OIL"]["Log_Return"]
    df["Gold_Return"] = market_data["GOLD"]["Log_Return"]
    df["VIX_Change"] = market_data["VIX"]["Close"].pct_change()

    # Calculate technical indicators for trend and volatility assessment
    df["Volatility_20"] = df["Log_Return"].rolling(20).std()
    df["Momentum_10"] = df["Close"] / df["Close"].shift(10) - 1
    df["Oil_Spike"] = df["Oil_Return"].rolling(5).mean()

    # Define target variable: predicted volatility regime for next day
    df["Target"] = (df["Volatility_20"].shift(-1) > df["Volatility_20"]).astype(int)

    # Remove rows with missing values from rolling window calculations
    df.dropna(inplace=True)

    return df