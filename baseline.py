# baseline.py - Logistic Regression Baseline Model
# Simple linear classifier to establish performance baseline

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_logistic_regression(df, features, target='Target', C=1.0):
    """
    Train Logistic Regression classifier.
    
    Mathematical Model:
    
    Logistic regression models the probability of class 1 as:
    
    P(y=1|X) = σ(β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ)
    
    Where:
    - σ(z) = 1/(1 + e^(-z))  # Sigmoid function
    - β = [β₀, β₁, ..., βₚ]  # Learned coefficients
    
    Decision Boundary:
    Predict 1 if P(y=1|X) > 0.5, else predict 0
    
    This corresponds to the linear decision boundary:
    β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ = 0
    
    Optimization:
    
    Minimize regularized negative log-likelihood:
    
    L(β) = -Σᵢ [yᵢ log(p̂ᵢ) + (1-yᵢ) log(1-p̂ᵢ)] + λ||β||²
    
    Where:
    - p̂ᵢ = P(yᵢ=1|Xᵢ; β)
    - λ = 1/C (regularization strength)
    - ||β||² = L2 penalty to prevent overfitting
    
    Interpretation:
    - Positive βⱼ: Feature Xⱼ increases risk probability
    - Negative βⱼ: Feature Xⱼ decreases risk probability
    - |βⱼ| magnitude indicates feature importance
    
    Args:
        df: DataFrame with features and target
        features: List of feature names
        target: Target column name
        C: Inverse regularization strength (higher = less regularization)
    
    Returns:
        model: Trained LogisticRegression model
        scaler: Fitted StandardScaler
    """
    # Extract features and target
    X = df[features]
    y = df[target]
    
    # Standardize features (required for regularized models)
    # Z-score normalization: X' = (X - μ) / σ
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train logistic regression with L2 regularization
    model = LogisticRegression(
        C=C,                      # Regularization strength
        penalty='l2',             # L2 penalty
        solver='lbfgs',           # Quasi-Newton optimizer
        max_iter=1000,            # Maximum iterations
        random_state=42
    )
    
    model.fit(X_scaled, y)
    
    return model, scaler


def get_coefficient_interpretation(model, scaler, feature_names):
    """
    Extract and interpret logistic regression coefficients.
    
    Coefficient Interpretation:
    
    For standardized features:
    βⱼ represents the change in log-odds for a 1 standard deviation
    increase in feature Xⱼ, holding all other features constant.
    
    Log-Odds Interpretation:
    log(P(y=1)/(1-P(y=1))) = β₀ + Σⱼ βⱼXⱼ
    
    Odds Ratio:
    OR(Xⱼ) = exp(βⱼ)
    - OR > 1: Feature increases odds of high-risk regime
    - OR < 1: Feature decreases odds of high-risk regime
    - OR = 1: Feature has no effect
    
    Returns:
        DataFrame with coefficients, odds ratios, and interpretations
    """
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    
    # Calculate odds ratios
    odds_ratios = np.exp(coefficients)
    
    # Create interpretation strings
    interpretations = []
    for coef, or_val in zip(coefficients, odds_ratios):
        if coef > 0:
            interp = f"1 SD increase → {or_val:.2f}x higher risk odds"
        else:
            interp = f"1 SD increase → {1/or_val:.2f}x lower risk odds"
        interpretations.append(interp)
    
    import pandas as pd
    results = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient (β)': coefficients,
        'Odds Ratio': odds_ratios,
        'Interpretation': interpretations
    })
    
    # Sort by absolute coefficient value
    results = results.reindex(
        results['Coefficient (β)'].abs().sort_values(ascending=False).index
    )
    
    return results, intercept


def logistic_predict_proba(model, scaler, X):
    """
    Generate probability predictions from logistic regression.
    
    Args:
        model: Trained LogisticRegression
        scaler: Fitted StandardScaler
        X: Features to predict on
    
    Returns:
        Array of probabilities P(y=1|X)
    """
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:, 1]
    return proba
