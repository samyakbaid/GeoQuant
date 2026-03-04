# app.py - Main Streamlit application for GeoQuant risk analysis system
# Three interactive modes: Live Risk Dashboard, Mathematical Framework, and Sector Rotation Monitor

import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import yfinance as yf

# Import custom modules for data processing and modeling
from data import get_market_data
from features import create_features
from model import train_model
from vol_model import garch_forecast

# Configure Streamlit page layout and styling
st.set_page_config(layout="wide")

st.title("GeoQuant — Geopolitical Risk Anticipation System")
st.markdown("Institutional Volatility & Cross-Asset Risk Modeling Framework")

# Select display mode from sidebar
mode = st.sidebar.selectbox(
    "System Mode",
    [
        "Live Risk Dashboard",
        "Mathematical Framework",
        "Sector Rotation Monitor"
    ]
)

if mode == "Live Risk Dashboard":
    # Load market data from Yahoo Finance for all asset classes
    data = get_market_data()
    # Create engineered features from raw market data
    df = create_features(data)
    # Train machine learning model on historical data
    model = train_model(df)

    # Define features used by the model
    features = [
        "Oil_Return",
        "Gold_Return",
        "VIX_Change",
        "Volatility_20",
        "Momentum_10",
        "Oil_Spike"
    ]

    # Get latest feature values for current prediction
    latest = df[features].iloc[-1:]

    # Generate risk probability from trained model
    risk_prob = model.predict_proba(latest)[0][1]

    # Forecast next-day volatility using GARCH model
    garch_vol = garch_forecast(df["Log_Return"])

    # Display key metrics
    st.metric("Predicted Risk Probability", f"{risk_prob:.2f}")
    st.metric("GARCH Forecast Volatility", f"{garch_vol:.2f}")

    # Plot rolling volatility with event annotations for major market events
    fig = px.line(x=df.index, y=df["Volatility_20"],
                  title="S&P 500 20-Day Volatility")

    st.plotly_chart(fig, use_container_width=True)

    # Display comparative performance across asset classes and risk factors
    st.subheader("Comparative Asset Performance")

    col1, col2 = st.columns(2)

    # Show correlation between commodity prices and equity volatility
    with col1:
        fig_assets = px.line(df, x=df.index, y=["Oil_Return", "Gold_Return"],
                            title="Oil vs Gold Returns",
                            labels={"value": "Return", "index": "Date"})
        fig_assets.update_layout(hovermode='x unified')
        st.plotly_chart(fig_assets, use_container_width=True)

    # Display volatility index as separate risk gauge
    with col2:
        fig_vix = px.line(x=df.index, y=df["VIX_Change"],
                         title="VIX Changes (Market Volatility Index)",
                         labels={"VIX_Change": "VIX Change", "index": "Date"})
        fig_vix.update_traces(line_color="red")
        st.plotly_chart(fig_vix, use_container_width=True)

    # Show all asset class returns on unified chart for comparative analysis
    st.subheader("Multi-Asset Returns Comparison")
    fig_all_returns = px.line(df, x=df.index, 
                              y=["Oil_Return", "Gold_Return", "VIX_Change"],
                              title="Comparative Asset Returns Over Time",
                              labels={"value": "Return/Change", "index": "Date"})
    fig_all_returns.update_layout(hovermode='x unified')
    st.plotly_chart(fig_all_returns, use_container_width=True)

    # Display price momentum as leading indicator of market direction
    st.subheader("Market Momentum")
    fig_momentum = px.line(x=df.index, y=df["Momentum_10"],
                          title="10-Day Price Momentum",
                          labels={"Momentum_10": "Momentum", "index": "Date"})
    fig_momentum.update_traces(line_color="green")
    st.plotly_chart(fig_momentum, use_container_width=True)


elif mode == "Mathematical Framework":
    # Section 1: Explain logarithmic return calculations
    st.header("1. Log Return Modeling")

    st.markdown("""
### Definition

Asset prices evolve multiplicatively:

P_t = P_{t-1} (1 + R_t)

To linearize this dynamic, we use logarithmic returns.
""")

    st.latex(r"""
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
""")

    st.markdown("""
### Why Log Returns?

1. Additivity over time:
""")

    st.latex(r"""
\ln\left(\frac{P_t}{P_{t-2}}\right)
=
\ln\left(\frac{P_t}{P_{t-1}}\right)
+
\ln\left(\frac{P_{t-1}}{P_{t-2}}\right)
""")

    st.markdown("""
2. Stabilizes variance
3. Approximates continuously compounded return

This is the return series used to compute:
• Rolling volatility
• GARCH conditional variance
• ML target variable
""")

    st.subheader("Numerical Example")

    # Demonstrate log return calculation with a concrete example
    price1 = 100
    price2 = 110

    log_ret = np.log(price2/price1)

    st.latex(r"\ln(110/100)")
    st.write(f"Log return = {log_ret:.4f}")

    # Section 2: Explain rolling volatility calculation
    st.header("2. Rolling Volatility Estimation")

    st.markdown("""
Volatility measures dispersion of returns.

For window size n:
""")

    st.latex(r"""
\sigma_t = \sqrt{\frac{1}{n} \sum_{i=1}^{n} r_{t-i}^2}
""")

    st.markdown("""
This is the empirical standard deviation of recent returns.

This dashboard displays:

• 20-day rolling volatility of S&P 500
• Used as input feature
• Used to define regime shift target
""")

    st.markdown("### Why This Matters")

    st.markdown("""
Volatility clustering is empirically observed in financial markets.

Periods of calm are followed by calm.
Periods of turbulence are followed by turbulence.

Detecting rising rolling volatility allows early detection of stress regimes.
""")

    st.subheader("Example Calculation")

    # Show practical example of volatility calculation
    example_returns = np.array([0.01, -0.02, 0.015, 0.005, -0.01])
    vol_example = np.sqrt(np.mean(example_returns**2))

    st.write(f"Example volatility = {vol_example:.4f}")

    # Section 3: Explain GARCH model for conditional volatility
    st.header("3. GARCH(1,1) Conditional Volatility Model")

    st.markdown("""
We model returns as:
""")

    st.latex(r"""
r_t = \sigma_t \epsilon_t
""")

    st.latex(r"""
\epsilon_t \sim N(0,1)
""")

    st.markdown("""
Conditional variance evolves as:
""")

    st.latex(r"""
\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2
""")

    st.markdown("""
Interpretation:

• ω → Long-run variance  
• α → Reaction to shocks  
• β → Volatility persistence  

Stationarity condition:
""")

    st.latex(r"""
\alpha + \beta < 1
""")

    st.markdown("""
This model captures volatility clustering — a stylized fact of financial markets.

The displayed "GARCH Forecast Volatility" on the main dashboard
is the one-step-ahead forecast:

""")

    st.latex(r"""
E[\sigma_{t+1}^2 | \mathcal{F}_t]
""")

    # Section 4: Explain volatility regime classification
    st.header("4. Volatility Regime Classification")

    st.markdown("""
We define the prediction target as:

1 if next-day volatility > current volatility  
0 otherwise
""")

    st.latex(r"""
Target_t =
\begin{cases}
1 & \text{if } \sigma_{t+1} > \sigma_t \\
0 & \text{otherwise}
\end{cases}
""")

    st.markdown("""
The Random Forest classifier estimates:

""")

    st.latex(r"""
P(Target_t = 1 | X_t)
""")

    st.markdown("""
Where X_t includes:

• Oil returns  
• Gold returns  
• VIX changes  
• Momentum  
• Rolling volatility  

This probability is displayed as:

"Predicted Risk Probability"
""")

    # Section 5: Explain cross-asset macro risk transmission
    st.header("5. Cross-Asset Macro Risk Transmission")

    st.markdown("""
Geopolitical tension affects markets through risk channels:

Oil ↑ → Supply risk  
Gold ↑ → Safe haven demand  
VIX ↑ → Implied volatility surge  
Equities ↓ → Risk-off positioning  

The model encodes these as leading indicators of volatility regime shift.
""")
    
    # Section 6: Explain softmax transformation for probability distribution
    st.header("6. Softmax Probability Transformation with Temperature Scaling")
    
    st.markdown("""
Softmax converts raw scores into calibrated probability distribution that sums to 1.
""")
    
    st.latex(r"""
P(class_i) = \frac{e^{s_i/T}}{\sum_j e^{s_j/T}}
""")
    
    st.markdown("""
Where:
• s_i is the score for class i
• T is temperature parameter (0 < T)
• Higher T makes distribution more uniform
• Lower T makes distribution more concentrated

Temperature scaling is crucial for sector rotation scoring:
• T = 1: Standard softmax (default behavior)
• T < 1: Sharper concentration on best sectors
• T > 1: Smoother distribution across sectors

This calibration ensures probabilities properly reflect sector momentum relative to S&P500.
""")

elif mode == "Sector Rotation Monitor":

    st.title("Sector Rotation & Momentum Intelligence System")
    st.markdown("Cross-Sectional Multi-Factor Model for Sector Leadership Forecasting")

    import yfinance as yf
    import numpy as np
    import pandas as pd
    import plotly.express as px

    # Define mapping between sector names and their ETF tickers
    # Each sector has a standard ETF that tracks its performance
    sector_map = {
        "Technology": "XLK",
        "Energy": "XLE",
        "Financials": "XLF",
        "Healthcare": "XLV",
        "Industrials": "XLI",
        "Consumer Discretionary": "XLY",
        "Utilities": "XLU",
        "Materials": "XLB"
    }

    # Extract list of ETF tickers for data download
    tickers = list(sector_map.values())

    # Download one year of historical price data for all sectors
    data = yf.download(
        tickers,
        period="1y",
        progress=False,
        auto_adjust=True
    )["Close"]

    # Validate that data was successfully downloaded
    if data.empty:
        st.error("Market data download failed.")
        st.stop()

    # Rename columns from ticker symbols to readable sector names
    inv_map = {v: k for k, v in sector_map.items()}
    data = data.rename(columns=inv_map)
    
    # Keep only expected sector columns to maintain consistency
    data = data[list(sector_map.keys())]

    # Calculate factor values for ranking and scoring
    # 20-Day Momentum: measures recent price appreciation relative to 20 days ago
    momentum_20 = data.pct_change(20)

    # 20-Day Volatility: rolling standard deviation of daily returns
    # Lower volatility is preferred (stable outperformance)
    volatility_20 = data.pct_change().rolling(20).std()

    # S&P 500 Benchmark: main market index for relative strength calculation
    sp500 = yf.download("^GSPC", period="1y", progress=False, auto_adjust=True)["Close"]
    # Ensure sp500 is a Series, not a DataFrame
    if isinstance(sp500, pd.DataFrame):
        sp500 = sp500.iloc[:, 0]
    sp500_ret_20 = sp500.pct_change(20)
    # Ensure sp500_ret_20 is a Series, not a DataFrame
    if isinstance(sp500_ret_20, pd.DataFrame):
        sp500_ret_20 = sp500_ret_20.iloc[:, 0]

    # Align sector and benchmark data to common date range
    momentum_20, sp500_ret_20 = momentum_20.align(sp500_ret_20, join="inner", axis=0)
    
   
    
    # Validate data alignment succeeded
    if momentum_20.empty or sp500_ret_20.empty:
        st.error("Sector and S&P500 data do not overlap. Check ticker symbols and date ranges.")
        with st.expander("Debug"):
            st.write(f"Momentum shape: {momentum_20.shape}")
            st.write(f"SP500 shape: {sp500_ret_20.shape}")


    # Calculate Relative Strength: sector outperformance vs market benchmark
    # Ensure sp500_ret_20 is broadcast correctly as a column
    if isinstance(sp500_ret_20, pd.Series):
        # Handle missing values in benchmark returns
        sp500_filled = sp500_ret_20.ffill().bfill()
        sp500_ret_20_values = sp500_filled.values.reshape(-1, 1)  # Shape: (n, 1)

        
        # Subtract benchmark momentum from sector momentum using broadcasting
        relative_strength = momentum_20.values - sp500_ret_20_values
        relative_strength = pd.DataFrame(relative_strength, index=momentum_20.index, columns=momentum_20.columns)
    else:
        relative_strength = momentum_20.sub(sp500_ret_20, axis=0)
    
    # Verify relative strength calculation completed

    
    # Align volatility data to match momentum and relative strength time range
    volatility_20 = volatility_20.loc[momentum_20.index]

    # Extract most recent values for each factor, handling NaN from rolling windows
    latest_momentum = momentum_20.bfill().iloc[-1]
    latest_relative_strength = relative_strength.bfill().iloc[-1]
    latest_volatility = volatility_20.bfill().iloc[-1]

    # Fill any remaining missing values with reasonable defaults
    latest_momentum = latest_momentum.fillna(0)
    latest_relative_strength = latest_relative_strength.fillna(0)
    latest_volatility = latest_volatility.fillna(latest_volatility[latest_volatility > 0].mean() if (latest_volatility > 0).any() else 0.01)
    
    # Validate we have complete factor data before proceeding
    if len(latest_momentum) == 0 or len(latest_relative_strength) == 0 or len(latest_volatility) == 0:
        st.error(" Unable to calculate factors — no valid data after alignment")
        with st.expander("Debug"):
            st.write("Latest Momentum:", latest_momentum)
            st.write("Latest RS:", latest_relative_strength)
            st.write("Latest Vol:", latest_volatility)

   
    with st.expander("SP500 Benchmark Return"):
        sp500_latest = sp500_ret_20.iloc[-1]
        if isinstance(sp500_latest, pd.Series):
            sp500_latest = sp500_latest.iloc[0] if len(sp500_latest) > 0 else 0
        st.write(f"Latest SP500 20-day return: {float(sp500_latest):.6f}")
        st.write("All SP500 returns (last 5 rows):")
        st.write(sp500_ret_20.tail(5))
        

    # Prepare feature matrix for scoring: latest values of momentum, relative strength, volatility
    features = pd.DataFrame({
        "Momentum": latest_momentum,
        "Relative_Strength": latest_relative_strength,
        "Volatility": latest_volatility
    })

    # Filter to only expected sectors for consistency
    valid_sectors = list(sector_map.keys())
    features = features.loc[valid_sectors]

    # Remove infinite values and fill missing data with feature averages
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.mean())
    
    # Display calculated feature values for verification
    with st.expander("Feature Values"):
        st.write("Feature DataFrame:")
        st.write(features)
        

    # Standardize features to z-scores for fair comparison across scales
    # Each factor now has mean=0 and std=1 for equal weighting
    features_z = (features - features.mean()) / (features.std(ddof=0) + 1e-8)

    # Invert volatility score (lower volatility is better, so negate it)
    features_z["Volatility"] = -features_z["Volatility"]

    # Compute composite score: weighted combination of factors
    # 50% weight on momentum, 40% on relative strength, 10% on volatility
    score = (
        0.5 * features_z["Momentum"] +
        0.4 * features_z["Relative_Strength"] +
        0.1 * features_z["Volatility"]
    )

    # Convert raw scores to probability distribution using softmax with temperature
    # Temperature parameter allows tuning of probability concentration
    temperature = st.sidebar.slider(
        "Softmax Temperature",
        0.2, 3.0, 0.8, 0.1
    )

    # Standardize scores for numerical stability in exponential calculation
    score_normalized = (score - score.mean()) / (score.std() + 1e-8)
    stable_score = score_normalized / temperature
    
    # Apply softmax: convert scores to normalized probability distribution
    exp_score = np.exp(stable_score)
    probability = exp_score / exp_score.sum()
    
    # Ensure probabilities are valid (between 0 and 1, sum to 1)
    probability = probability.clip(0, 1)
    probability = probability / probability.sum()  # Re-normalize after clipping
    
    # Debug output showing all calculation steps
    with st.expander("Scores & Probabilities"):
        debug_df = pd.DataFrame({
            "Raw Score": score,
            "Normalized Score": score_normalized,
            "Stable Score": stable_score,
            "Probability": probability
        })
        st.write(debug_df)

    # Identify current best performer and predicted future leader
    booming_sector_now = latest_momentum.idxmax()
    predicted_future_leader = probability.idxmax()

    # Display key findings in summary format
    st.markdown("### Sector Leadership Summary")

    st.success(f"Current Momentum Leader: **{booming_sector_now}**")
    st.info(f"Highest 30-Day Forward Probability: **{predicted_future_leader}**")

    st.markdown("---")

    # Display results as bar charts and summary table
    st.subheader("20-Day Momentum Ranking")

    # Show sector momentum in descending order
    fig1 = px.bar(
        latest_momentum.sort_values(ascending=False),
        title="Sector 20-Day Momentum"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Forward 30-Day Outperformance Probability")

    # Sort sectors by probability and convert to percentages
    prob_sorted = probability.sort_values(ascending=False)
    prob_pct = (prob_sorted * 100).round(2)
    
    # Create bar chart with probability values as text labels
    fig2 = px.bar(
        x=prob_sorted.index,
        y=prob_pct.values,
        title="Softmax Probability Distribution (%)",
        labels={"x": "Sector", "y": "Probability (%)"},
        text=[f"{p:.2f}%" for p in prob_pct.values]
    )

    fig2.update_layout(
        yaxis_range=[0, max(20, prob_pct.max() * 1.2)],
        showlegend=False,
        height=400,
        xaxis_tickangle=-45
    )
    fig2.update_traces(textposition="outside", textangle=0)

    st.plotly_chart(fig2, use_container_width=True)
    
    # Display probability results in table format for precise viewing
    st.dataframe(
        pd.DataFrame({
            "Sector": prob_sorted.index,
            "Probability": (prob_sorted * 100).round(3).astype(str) + "%"
        }).reset_index(drop=True),
        use_container_width=True
    )

    st.markdown("---")

    # Explain mathematical foundation of the sector rotation model
    st.header("Mathematical Derivation of Sector Rotation Model")

    st.markdown("## 1. Price Return Definition")

    st.latex(r"""
    R_{i,t}^{(k)} =
    \frac{P_{i,t} - P_{i,t-k}}{P_{i,t-k}}
    """)

    st.markdown("""
Where:

P_{i,t} = price of sector i at time t  
k = 20 trading days  

This measures percentage capital appreciation over one month.
""")

    st.markdown("## 2. Momentum Factor")

    st.latex(r"""
    M_{i,t} =
    \frac{P_{i,t}}{P_{i,t-20}} - 1
    """)

    st.markdown("""
Momentum captures trend persistence.

Economic reasoning:
Assets that recently outperformed tend to continue outperforming
due to capital flow inertia and institutional positioning.
""")

    st.markdown("## 3. Relative Strength")

    st.latex(r"""
    RS_{i,t} =
    M_{i,t} - M_{m,t}
    """)

    st.markdown("""
Where M_{m,t} is the S&P 500 20-day return.

If RS > 0:
Sector is outperforming the broad market.

This isolates sector-specific alpha.
""")

    st.markdown("## 4. Volatility Estimation")

    st.latex(r"""
    \sigma_{i,t}^{(20)} =
    \sqrt{
    \frac{1}{20}
    \sum_{j=1}^{20}
    r_{i,t-j}^2
    }
    """)

    st.markdown("""
Volatility measures dispersion of daily returns.

Higher volatility implies greater uncertainty.
We penalize high volatility sectors for risk adjustment.
""")

    st.markdown("## 5. Cross-Sectional Standardization")

    st.latex(r"""
    Z_i =
    \frac{X_i - \mu_X}{\sigma_X}
    """)

    st.markdown("""
Convert all factors to z-scores using cross-sectional statistics.

This ensures:

• Zero mean  
• Unit variance  
• Comparable factor magnitudes
""")

    st.markdown("## 6. Composite Score")

    st.latex(r"""
    Score_i =
    0.5 Z_i^{(Momentum)}
    +
    0.4 Z_i^{(RelativeStrength)}
    +
    0.1 Z_i^{(Volatility)}
    """)

    st.markdown("""
Weighted combination of standardized factors.

Weight allocation reflects economic importance:

50% trend momentum  
40% market outperformance  
10% risk penalty
""")

    st.markdown("## 7. Softmax Probability Transformation")

    st.latex(r"""
    P_i =
    \frac{
    \exp\left(\frac{Score_i - \max_j Score_j}{T}\right)
    }{
    \sum_k
    \exp\left(\frac{Score_k - \max_j Score_j}{T}\right)
    }
    """)

    st.markdown("""
Convert scores to probability distribution.

Properties:

• 0 < P_i < 1  
• Sum_i P_i = 1  
• Higher score → exponentially higher probability  

T = temperature parameter controlling probability concentration.

Lower T sharpens concentration on best sectors  
Higher T smooths distribution across all sectors
""")

    st.markdown("## Interpretation")

    st.markdown("""
If Technology has probability 0.42:

It means that, relative to other sectors,
Technology has the highest predicted future leadership likelihood
based on current momentum and strength metrics.

This is a cross-sectional probabilistic ranking model,
not an absolute market probability forecast.
""")