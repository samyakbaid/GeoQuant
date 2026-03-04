# model.py - Train Random Forest classifier for risk prediction
# Uses cross-asset and technical features to predict volatility regime changes

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(df):
    # List of features used for prediction
    # Includes commodity returns, volatility measures, and momentum indicators
    features = [
        "Oil_Return",
        "Gold_Return",
        "VIX_Change",
        "Volatility_20",
        "Momentum_10",
        "Oil_Spike"
    ]

    # Extract feature matrix and target labels
    X = df[features]
    y = df["Target"]

    # Split data chronologically (shuffle=False) to respect time series nature
    # Use 80% for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )

    # Train Random Forest with 300 trees for robust predictions
    model = RandomForestClassifier(n_estimators=300)
    model.fit(X_train, y_train)

    return model