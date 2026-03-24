# app_enhanced.py - Enhanced GeoQuant Application with LSTM and Evaluation
# Adds: LSTM deep learning, walk-forward validation, model comparison, and comprehensive metrics

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import yfinance as yf

# Import original modules
from data import get_market_data
from features import create_features
from model import train_model
from vol_model import garch_forecast

# Import new modules
from baseline import train_logistic_regression, get_coefficient_interpretation
from lstm_model import train_lstm_model, lstm_predict
from evaluation import (
    walk_forward_validation,
    plot_confusion_matrix,
    plot_roc_curve,
    compare_models,
    get_classification_metrics,
    feature_importance_plot
)

# Configure page
st.set_page_config(layout="wide")

st.title("GeoQuant — Geopolitical Risk Anticipation System")
st.markdown("**Enhanced ML Framework: Classical Econometrics + Tree-Based Models + Deep Learning**")

# Sidebar mode selection
mode = st.sidebar.selectbox(
    "System Mode",
    [
        "Live Risk Dashboard",
        "Model Comparison & Evaluation",
        "Mathematical Framework",
        "Sector Rotation Monitor"
    ]
)

# Cache data loading
@st.cache_data
def load_and_prepare_data():
    """Load market data and create features (cached)"""
    data = get_market_data()
    df = create_features(data)
    return df

if mode == "Live Risk Dashboard":
    st.header("Real-Time Risk Prediction Dashboard")
    
    # Load data
    df = load_and_prepare_data()
    
    # Define features
    features = [
        "Oil_Return",
        "Gold_Return",
        "VIX_Change",
        "Volatility_20",
        "Momentum_10",
        "Oil_Spike"
    ]
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Prediction Model",
        ["Random Forest", "Logistic Regression", "LSTM Neural Network"]
    )
    
    latest = df[features].iloc[-1:]
    
    if model_choice == "Random Forest":
        model = train_model(df)
        risk_prob = model.predict_proba(latest)[0][1]
        
    elif model_choice == "Logistic Regression":
        from sklearn.model_selection import train_test_split
        X = df[features]
        y = df["Target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, shuffle=False, test_size=0.2
        )
        
        lr_model, lr_scaler = train_logistic_regression(df, features)
        risk_prob = lr_model.predict_proba(lr_scaler.transform(latest))[0][1]
        
    else:  # LSTM
        st.info("Training LSTM model (this may take a minute)...")
        
        # Train LSTM with reduced epochs for speed
        lstm_model, lstm_scaler, history = train_lstm_model(
            df, features, target='Target', 
            lookback=21, epochs=30, batch_size=32
        )
        
        # Predict with LSTM
        risk_prob = lstm_predict(lstm_model, lstm_scaler, df, features, lookback=21)
        
        # Show training history
        with st.expander("LSTM Training History"):
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=history.history['loss'],
                mode='lines',
                name='Training Loss'
            ))
            fig_loss.add_trace(go.Scatter(
                y=history.history['val_loss'],
                mode='lines',
                name='Validation Loss'
            ))
            fig_loss.update_layout(
                title='LSTM Training & Validation Loss',
                xaxis_title='Epoch',
                yaxis_title='Binary Cross-Entropy Loss'
            )
            st.plotly_chart(fig_loss, use_container_width=True)
    
    # GARCH forecast
    garch_vol = garch_forecast(df["Log_Return"])
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Risk Probability", f"{risk_prob:.2%}")
    with col2:
        st.metric("GARCH Forecast Volatility", f"{garch_vol:.4f}")
    
    # Volatility plot
    fig = px.line(x=df.index, y=df["Volatility_20"],
                  title="S&P 500 20-Day Rolling Volatility")
    st.plotly_chart(fig, use_container_width=True)
    
    # Asset performance
    st.subheader("Cross-Asset Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_assets = px.line(df, x=df.index, y=["Oil_Return", "Gold_Return"],
                            title="Commodity Returns: Oil vs Gold")
        st.plotly_chart(fig_assets, use_container_width=True)
    
    with col2:
        fig_vix = px.line(x=df.index, y=df["VIX_Change"],
                         title="VIX Changes (Market Fear Gauge)")
        fig_vix.update_traces(line_color="red")
        st.plotly_chart(fig_vix, use_container_width=True)

elif mode == "Model Comparison & Evaluation":
    st.header("Comprehensive Model Evaluation & Comparison")
    
    st.markdown("""
    This section implements rigorous walk-forward validation to compare:
    1. **Logistic Regression** (Linear Baseline)
    2. **Random Forest** (Tree-Based Ensemble)
    3. **LSTM** (Sequential Deep Learning)
    """)
    
    # Load data
    df = load_and_prepare_data()
    
    features = [
        "Oil_Return",
        "Gold_Return",
        "VIX_Change",
        "Volatility_20",
        "Momentum_10",
        "Oil_Spike"
    ]
    
    # Validation settings
    n_splits = st.sidebar.slider("Number of Folds", 3, 10, 5)
    
    if st.button("Run Walk-Forward Validation"):
        with st.spinner("Running cross-validation..."):
            
            # Define model builders
            def build_lr():
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import Pipeline
                return Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(C=1.0, max_iter=1000))
                ])
            
            def build_rf():
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=300, random_state=42)
            
            # Run walk-forward validation
            st.subheader("1. Logistic Regression")
            lr_results, lr_preds = walk_forward_validation(
                df, features, 'Target', build_lr, n_splits
            )
            st.dataframe(lr_results)
            
            st.subheader("2. Random Forest")
            rf_results, rf_preds = walk_forward_validation(
                df, features, 'Target', build_rf, n_splits
            )
            st.dataframe(rf_results)
            
            # LSTM validation (simplified - single fold due to computational cost)
            st.subheader("3. LSTM Neural Network")
            st.info("Training LSTM (single fold for computational efficiency)...")
            
            from sklearn.model_selection import train_test_split
            X = df[features]
            y = df['Target']
            
            # Chronological split
            split_idx = int(0.8 * len(df))
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
            
            lstm_model, lstm_scaler, _ = train_lstm_model(
                train_df, features, target='Target',
                lookback=21, epochs=30, batch_size=32, validation_split=0.2
            )
            
            # Evaluate on test set
            from lstm_model import create_sequences
            test_scaled = test_df.copy()
            test_scaled[features] = lstm_scaler.transform(test_df[features])
            
            X_seq_test, y_seq_test = create_sequences(
                test_scaled, features, 'Target', lookback=21
            )
            
            y_pred_proba = lstm_model.predict(X_seq_test, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            lstm_metrics = get_classification_metrics(
                y_seq_test, y_pred, y_pred_proba
            )
            
            st.write("LSTM Test Set Metrics:")
            st.json(lstm_metrics)
            
            # Model comparison
            st.header("Model Comparison")
            
            # Average metrics across folds
            st.subheader("Average Performance Metrics")
            
            comparison = pd.DataFrame({
                'Model': ['Logistic Regression', 'Random Forest', 'LSTM'],
                'Accuracy': [
                    lr_results['accuracy'].mean(),
                    rf_results['accuracy'].mean(),
                    lstm_metrics['accuracy']
                ],
                'Precision': [
                    lr_results['precision'].mean(),
                    rf_results['precision'].mean(),
                    lstm_metrics['precision']
                ],
                'Recall': [
                    lr_results['recall'].mean(),
                    rf_results['recall'].mean(),
                    lstm_metrics['recall']
                ],
                'F1 Score': [
                    lr_results['f1'].mean(),
                    rf_results['f1'].mean(),
                    lstm_metrics['f1']
                ],
                'AUC-ROC': [
                    lr_results['auc_roc'].mean(),
                    rf_results['auc_roc'].mean(),
                    lstm_metrics['auc_roc']
                ]
            })
            
            st.dataframe(comparison)
            
            # Comparative plot
            fig_comp = compare_models({
                'Logistic Regression': lr_results,
                'Random Forest': rf_results
            })
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Confusion matrices
            st.header("Confusion Matrices")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_cm_lr = plot_confusion_matrix(
                    lr_preds['y_true'], 
                    lr_preds['y_pred'],
                    "Logistic Regression"
                )
                st.plotly_chart(fig_cm_lr, use_container_width=True)
            
            with col2:
                fig_cm_rf = plot_confusion_matrix(
                    rf_preds['y_true'], 
                    rf_preds['y_pred'],
                    "Random Forest"
                )
                st.plotly_chart(fig_cm_rf, use_container_width=True)
            
            # LSTM confusion matrix
            fig_cm_lstm = plot_confusion_matrix(
                y_seq_test, y_pred, "LSTM Neural Network"
            )
            st.plotly_chart(fig_cm_lstm, use_container_width=True)
            
            # ROC curves
            st.header("ROC Curves")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_roc_lr = plot_roc_curve(
                    lr_preds['y_true'],
                    lr_preds['y_proba'],
                    "Logistic Regression ROC"
                )
                st.plotly_chart(fig_roc_lr, use_container_width=True)
            
            with col2:
                fig_roc_rf = plot_roc_curve(
                    rf_preds['y_true'],
                    rf_preds['y_proba'],
                    "Random Forest ROC"
                )
                st.plotly_chart(fig_roc_rf, use_container_width=True)
            
            # Feature importance (Random Forest)
            st.header("Feature Importance Analysis")
            
            # Train a final RF model for feature importance
            from sklearn.ensemble import RandomForestClassifier
            final_rf = RandomForestClassifier(n_estimators=300, random_state=42)
            final_rf.fit(df[features], df['Target'])
            
            fig_fi = feature_importance_plot(final_rf, features)
            st.plotly_chart(fig_fi, use_container_width=True)
            
            # Logistic regression coefficients
            st.subheader("Logistic Regression Coefficient Interpretation")
            
            lr_model, lr_scaler = train_logistic_regression(df, features)
            coef_df, intercept = get_coefficient_interpretation(
                lr_model, lr_scaler, features
            )
            
            st.dataframe(coef_df)
            
            st.markdown(f"""
            **Intercept (β₀):** {intercept:.4f}
            
            **Interpretation:** When all features are at their mean values (after standardization),
            the log-odds of a high-risk regime is {intercept:.4f}.
            """)

elif mode == "Mathematical Framework":
    st.header("Mathematical & Machine Learning Framework")
    st.markdown("""
    This system bridges classical financial econometrics with modern machine learning. 
    The architecture is divided into four sequential phases: **Target Generation, Static Classification, Sequential Deep Learning, and Sector Ranking.**
    """)

    st.markdown("---")

    # PHASE 1
    st.header("Phase 1: Ground Truth Target Generation (GARCH)")
    st.markdown("""
    Machine learning classifiers require discrete target labels. Because true market volatility is unobservable, 
    we extract the conditional variance using a classical Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model.
    
    First, asset prices are converted to logarithmic returns to ensure additivity and stationarity:
    """)
    st.latex(r"r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)")
    
    st.markdown("The **GARCH(1,1)** conditional variance $\sigma_t^2$ evolves as:")
    st.latex(r"\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2")
    
    st.markdown("""
    Where:
    * **$\omega$**: Long-run baseline variance
    * **$\alpha r_{t-1}^2$**: ARCH term (reaction to recent market shocks)
    * **$\beta \sigma_{t-1}^2$**: GARCH term (persistence of past volatility)
    
    Using Maximum Likelihood Estimation (MLE), we calculate the one-step-ahead forecasted volatility ($\sigma_{t+1}$). 
    The binary classification target ($y_t$) is then generated using a logical step function, classifying tomorrow as a higher-risk regime (1) or lower-risk regime (0):
    """)
    st.latex(r"""
    y_t = \begin{cases} 
    1 & \text{if } \sigma_{t+1} > \sigma_t \\ 
    0 & \text{otherwise} 
    \end{cases}
    """)

    st.markdown("---")

    # PHASE 2
    st.header("Phase 2: Static Regime Classification (Baselines)")
    st.markdown("""
    With targets defined, we establish baselines using models that evaluate a static "snapshot" of macroeconomic features $X_t$ (Oil, Gold, VIX, etc.).
    """)

    st.subheader("Logistic Regression (Linear Baseline)")
    st.markdown("Models the log-odds of a high-risk regime as a linear combination of features, mapped to a probability between 0 and 1 using the sigmoid function:")
    st.latex(r"P(y_t = 1 | X_t) = \frac{1}{1 + e^{-(\beta_0 + \sum_{i=1}^n \beta_i x_{i,t})}}")

    st.subheader("Random Forest (Tree-Based Ensemble)")
    st.markdown("""
    Captures non-linear relationships without easily overfitting by aggregating predictions from $B$ decision trees. 
    Splits are determined by minimizing Gini Impurity ($H$):
    """)
    st.latex(r"H(Q) = \sum_{k \in \{0, 1\}} p_k (1 - p_k)")
    st.markdown("The final probability is the averaged vote across all trees:")
    st.latex(r"P(y_t = 1 | X_t) = \frac{1}{B} \sum_{b=1}^B \hat{y}_b(X_t)")

    st.markdown("---")

    # PHASE 3
    st.header("Phase 3: Sequential Deep Learning (LSTM)")
    st.markdown("""
    Financial markets are inherently sequential; the momentum of the past month influences tomorrow's risk. 
    To capture this chronological trajectory, we restructure the 2D feature matrix into 3D rolling windows (e.g., 21-day sequences) and pass them through a Long Short-Term Memory (LSTM) network.
    
    The LSTM maintains an internal cell state ($C_t$) regulated by three distinct gates:
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**1. Forget Gate:** Decides what past information to discard.")
        st.latex(r"f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)")
        
        st.markdown("**2. Input Gate:** Decides what new information to store.")
        st.latex(r"i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)")
        st.latex(r"\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)")

    with col2:
        st.markdown("**3. Cell State Update:** Merges past memory with new inputs.")
        st.latex(r"C_t = f_t * C_{t-1} + i_t * \tilde{C}_t")
        
        st.markdown("**4. Output Gate:** Determines the next hidden state.")
        st.latex(r"o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)")
        st.latex(r"h_t = o_t * \tanh(C_t)")

    st.markdown("""
    The final hidden state is passed through a dense layer with a sigmoid activation to output the sequence-based probability of a regime shift.
    """)

    st.markdown("---")

    # PHASE 4
    st.header("Phase 4: Probabilistic Sector Ranking")
    st.markdown("""
    To make risk forecasts actionable, sectors are evaluated cross-sectionally. Raw metrics (Momentum, Relative Strength, Volatility) are standardized into Z-scores:
    """)
    st.latex(r"Z_{i, factor} = \frac{X_i - \mu_X}{\sigma_X}")
    
    st.markdown("A weighted composite score is calculated to favor momentum while penalizing volatility:")
    st.latex(r"Score_i = 0.5 Z_i^{Momentum} + 0.4 Z_i^{RelativeStrength} + 0.1 Z_i^{Volatility*}")
    st.caption("*Note: Volatility Z-score is inverted so lower risk yields a higher score.")

    st.markdown("""
    Finally, a **Softmax Temperature Transformation** squashes these unbounded scores into a dynamic, cross-sectional probability distribution:
    """)
    st.latex(r"P_i = \frac{\exp\left(\frac{Score_i - \max(Score)}{T}\right)}{\sum_j \exp\left(\frac{Score_j - \max(Score)}{T}\right)}")
    
    st.markdown("""
    Where **$T$ (Temperature)** controls distribution concentration:
    * $T = 1$: Standard distribution.
    * $T < 1$: Sharper concentration on the highest-scoring sectors.
    * $T > 1$: Smoother, more uniform distribution across all sectors.
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
