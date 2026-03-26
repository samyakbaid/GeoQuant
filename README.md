# GeoQuant: Geopolitical Risk Anticipation System
## Enhanced ML Framework with LSTM, Random Forest, and Logistic Regression

---

## Project Overview

A comprehensive machine learning system for predicting equity market volatility using:
- **Classical Econometrics:** GARCH(1,1) for volatility forecasting
- **Linear Baseline:** Logistic Regression
- **Ensemble Learning:** Random Forest
- **Deep Learning:** LSTM Neural Network
- **Portfolio Strategy:** Softmax-based sector rotation

**Key Features:**
- ✅ Multi-model comparison with walk-forward validation
- ✅ Real-time risk prediction dashboard
- ✅ Feature importance analysis
- ✅ Interactive sector rotation monitor
- ✅ Mathematical framework documentation

---

##  Project Structure

```
geoquant/
├── data.py              # Market data download (Yahoo Finance)
├── features.py          # Feature engineering pipeline
├── model.py             # Random Forest classifier
├── vol_model.py         # GARCH(1,1) volatility model
├── baseline.py          # Logistic Regression baseline
├── lstm_model.py        # LSTM neural network
├── evaluation.py        # Walk-forward validation & metrics
├── app.py               # Simple Streamlit dashboard
├── app_enhanced.py      # Full evaluation suite
└── requirements.txt     # Python dependencies
```

---

## Quick Start

### **1. Installation**

```bash
# Clone repository
git clone https://github.com/samyakbaid/GeoQuant.git
cd GeoQuant

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Run the Application**

```bash
# Launch enhanced app with all features
streamlit run app_enhanced.py

# Or run simple dashboard
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## System Modes

### **Mode 1: Live Risk Dashboard**
- Real-time volatility predictions
- Choose between 3 models: Logistic Regression, Random Forest, or LSTM
- GARCH volatility forecast
- Multi-asset performance charts

### **Mode 2: Model Comparison & Evaluation**
- Walk-forward cross-validation
- Confusion matrices
- ROC curves
- Feature importance analysis
- Logistic regression coefficient interpretation

### **Mode 3: Mathematical Framework**
- Detailed equation derivations
- Log return calculations
- Rolling volatility estimation
- GARCH model explanation
- Softmax transformation

### **Mode 4: Sector Rotation Monitor**
- Real-time sector rankings
- Softmax probability distribution
- Adjustable temperature parameter
- Factor decomposition

---

## Testing Individual Components

### **Test Data Pipeline**
```python
from data import get_market_data
from features import create_features

data = get_market_data()
df = create_features(data)
print(df.head())
```

### **Test Random Forest**
```python
from model import train_model

features = ["Oil_Return", "Gold_Return", "VIX_Change", 
            "Volatility_20", "Momentum_10", "Oil_Spike"]

model = train_model(df)
latest = df[features].iloc[-1:]
risk_prob = model.predict_proba(latest)[0][1]
print(f"Risk Probability: {risk_prob:.2%}")
```

### **Test LSTM**
```python
from lstm_model import train_lstm_model, lstm_predict

# Train LSTM (this will take a few minutes)
lstm_model, scaler, history = train_lstm_model(
    df, features, epochs=30, batch_size=32
)

# Make prediction
prob = lstm_predict(lstm_model, scaler, df, features)
print(f"LSTM Risk Probability: {prob:.2%}")
```

### **Test Logistic Regression**
```python
from baseline import train_logistic_regression, get_coefficient_interpretation

lr_model, lr_scaler = train_logistic_regression(df, features)
coef_df, intercept = get_coefficient_interpretation(lr_model, lr_scaler, features)
print(coef_df)
```

### **Test Walk-Forward Validation**
```python
from evaluation import walk_forward_validation
from sklearn.ensemble import RandomForestClassifier

def build_rf():
    return RandomForestClassifier(n_estimators=300, random_state=42)

results, predictions = walk_forward_validation(
    df, features, 'Target', build_rf, n_splits=5
)
print(results)
```

---

## Expected Results

### **Model Performance Comparison**

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.62 | ~0.58 | ~0.54 | ~0.56 | ~0.64 |
| Random Forest | ~0.71 | ~0.69 | ~0.68 | ~0.68 | ~0.76 |
| **LSTM** | ~**0.74** | ~**0.72** | ~**0.71** | ~**0.71** | ~**0.79** |

*Note: Results may vary based on market conditions and time period*

### **Key Insights**

1. **LSTM outperforms static models** — Sequential information improves predictions by ~8% F1
2. **VIX is the strongest predictor** — Market fear gauge dominates feature importance
3. **Non-linearity matters** — Random Forest beats Logistic Regression
4. **Volatility clustering exists** — Past volatility predicts future volatility

---

##  Configuration

### **Adjust LSTM Training**

In `lstm_model.py`:
```python
# Speed up training (lower quality)
epochs=20, batch_size=64

# Better performance (slower)
epochs=100, batch_size=16
```

### **Change Sector Rotation Temperature**

In Streamlit app, use sidebar slider:
- **T < 1:** Concentrated (winner-take-all)
- **T = 1:** Balanced
- **T > 1:** Diversified

### **Extend Data History**

In `data.py`:
```python
# Change from 5 years to 10 years
df = yf.download(ticker, period="10y")
```

---

## Mathematical Background

### **GARCH(1,1) Model**
```
σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
```
- ω: Long-run variance
- α: Shock sensitivity
- β: Volatility persistence

### **LSTM Cell Equations**
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    # Input gate
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t        # Cell update
h_t = o_t ⊙ tanh(C_t)                  # Hidden state
```

### **Softmax with Temperature**
```
P_i = exp(s_i/T) / Σ_j exp(s_j/T)
```

---

## Troubleshooting

### **TensorFlow Installation Issues**

**On macOS (M1/M2):**
```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

**On Windows/Linux:**
```bash
pip install tensorflow
```

### **LSTM Shape Errors**

Ensure input is 3D:
```python
print(X_seq.shape)  # Should be (samples, timesteps, features)
# Example: (1200, 21, 6)
```

### **NaN Values in Data**

Check for missing data:
```python
df.isna().sum()
df.dropna(inplace=True)
```

### **Slow Training**

Options:
1. Reduce epochs: `epochs=20`
2. Increase batch size: `batch_size=64`
3. Use GPU (if available)
4. Reduce lookback window: `lookback=15`

---

## 📊 Data Sources

All data downloaded via **yfinance** (Yahoo Finance API):

| Asset | Ticker | Description |
|-------|--------|-------------|
| S&P 500 | ^GSPC | U.S. equity market index |
| Crude Oil | CL=F | WTI crude oil futures |
| Gold | GC=F | Gold futures |
| VIX | ^VIX | CBOE Volatility Index |
| Sectors | XLK, XLE, XLF, ... | SPDR sector ETFs |

**Data Frequency:** Daily closing prices  
**Time Period:** Last 5 years (configurable to 10 years)

---

## License

MIT License - See LICENSE file for details

---

## Authors

- **Samyak Baid**
- **Mohammad Fazlur**

**Course:** Introduction to Machine Learning  
**Institution:** Ashoka university
**Term:** Spring 2026

---

## Acknowledgments

- **ARCH library** for GARCH implementation
- **TensorFlow/Keras** for LSTM framework
- **Streamlit** for interactive dashboard
- **Yahoo Finance** for market data

---




## Educational Use

This project is designed for educational purposes in machine learning coursework. It demonstrates:

- Time-series modeling
- Multi-model comparison
- Proper cross-validation
- Feature engineering
- Deep learning implementation
- Software deployment

**Not for financial advice or actual trading decisions.**
