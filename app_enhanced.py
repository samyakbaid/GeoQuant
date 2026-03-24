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
    # [Keep the existing Mathematical Framework code from original app.py]
    # This section remains unchanged
    pass  # Original code goes here

elif mode == "Sector Rotation Monitor":
    # [Keep the existing Sector Rotation Monitor code from original app.py]
    # This section remains unchanged
    pass  # Original code goes here
