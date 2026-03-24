# lstm_model.py - LSTM Neural Network for Sequential Volatility Prediction
# Captures temporal dependencies in market dynamics using deep learning

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

def create_sequences(data, features, target, lookback=21):
    """
    Convert tabular data into sequential format for LSTM.
    
    Mathematical Formulation:
    For each time t, create sequence:
    X_t = [X_{t-lookback+1}, X_{t-lookback+2}, ..., X_t]
    y_t = Target_t
    
    Args:
        data: DataFrame with features and target
        features: List of feature column names
        target: Target column name
        lookback: Number of historical days to include (default 21 trading days ≈ 1 month)
    
    Returns:
        X_seq: 3D array (samples, timesteps, features)
        y_seq: 1D array of targets
    """
    X = data[features].values
    y = data[target].values
    
    X_seq = []
    y_seq = []
    
    # Create rolling windows
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])  # Past 'lookback' days
        y_seq.append(y[i])              # Target for day i
    
    return np.array(X_seq), np.array(y_seq)


def build_lstm_model(input_shape, units=[64, 32]):
    """
    Build LSTM architecture for binary classification.
    
    Architecture:
    1. LSTM Layer 1: 64 units with return_sequences=True (stacked LSTM)
    2. Dropout: 20% regularization to prevent overfitting
    3. LSTM Layer 2: 32 units
    4. Dropout: 20%
    5. Dense Output: 1 unit with sigmoid activation
    
    Mathematical Model:
    
    LSTM Cell Update Equations:
    
    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)           # Forget gate
    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)           # Input gate
    C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)        # Candidate cell state
    C_t = f_t * C_{t-1} + i_t * C̃_t               # Cell state update
    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)           # Output gate
    h_t = o_t * tanh(C_t)                         # Hidden state
    
    Where:
    - σ = sigmoid activation
    - * = element-wise multiplication
    - W, b = learnable weight matrices and biases
    
    Output Layer:
    P(y=1|X) = σ(W_out · h_T + b_out)
    
    Args:
        input_shape: Tuple (timesteps, num_features)
        units: List of LSTM layer sizes
    
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # First LSTM layer with return_sequences=True for stacking
    model.add(LSTM(
        units[0], 
        return_sequences=True, 
        input_shape=input_shape,
        name='LSTM_Layer_1'
    ))
    model.add(Dropout(0.2, name='Dropout_1'))
    
    # Second LSTM layer
    model.add(LSTM(units[1], name='LSTM_Layer_2'))
    model.add(Dropout(0.2, name='Dropout_2'))
    
    # Output layer with sigmoid for binary classification
    model.add(Dense(1, activation='sigmoid', name='Output_Layer'))
    
    # Compile with Adam optimizer and binary cross-entropy loss
    # Loss function: L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_lstm_model(df, features, target='Target', lookback=21, 
                     epochs=50, batch_size=32, validation_split=0.2):
    """
    Train LSTM model with proper time-series validation.
    
    Time-Series Split:
    |----Training----|----Validation----|----Test----|
    
    Args:
        df: DataFrame with features and target
        features: List of feature names
        target: Target column name
        lookback: Sequence length
        epochs: Training iterations
        batch_size: Mini-batch size for SGD
        validation_split: Fraction for validation (chronological split)
    
    Returns:
        model: Trained Keras model
        scaler: Fitted StandardScaler for features
        history: Training history
    """
    # Standardize features (required for neural networks)
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    
    # Create sequences
    X_seq, y_seq = create_sequences(df_scaled, features, target, lookback)
    
    print(f"Sequence shape: {X_seq.shape}")
    print(f"Target shape: {y_seq.shape}")
    
    # Build model
    model = build_lstm_model(input_shape=(lookback, len(features)))
    
    print("\nModel Architecture:")
    model.summary()
    
    # Train with early stopping awareness
    # Use validation_split to monitor overfitting
    history = model.fit(
        X_seq, y_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
        shuffle=False  # CRITICAL: No shuffling for time series
    )
    
    return model, scaler, history


def lstm_predict(model, scaler, df, features, lookback=21):
    """
    Generate predictions using trained LSTM.
    
    Args:
        model: Trained Keras model
        scaler: Fitted StandardScaler
        df: Recent data for prediction
        features: Feature names
        lookback: Sequence length
    
    Returns:
        Probability of regime shift
    """
    # Standardize latest data
    df_scaled = df.copy()
    df_scaled[features] = scaler.transform(df[features])
    
    # Extract last 'lookback' days
    X_latest = df_scaled[features].tail(lookback).values
    X_latest = X_latest.reshape(1, lookback, len(features))
    
    # Predict probability
    prob = model.predict(X_latest, verbose=0)[0][0]
    
    return prob
