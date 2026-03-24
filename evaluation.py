# evaluation.py - Comprehensive Model Evaluation Framework
# Implements walk-forward validation and classification metrics

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_curve,
    roc_auc_score,
    accuracy_score
)
from sklearn.model_selection import TimeSeriesSplit

def walk_forward_validation(df, features, target, model_builder, n_splits=5):
    """
    Implement walk-forward cross-validation for time series.
    
    Mathematical Formulation:
    
    Split data chronologically into n_splits folds:
    
    Fold 1: Train[0:t1] → Test[t1:t2]
    Fold 2: Train[0:t2] → Test[t2:t3]
    ...
    Fold k: Train[0:tk] → Test[tk:tk+1]
    
    This ensures:
    1. No look-ahead bias (model never sees future data)
    2. Expanding training window (more realistic for financial data)
    3. Consistent evaluation across time regimes
    
    Args:
        df: DataFrame with features and target
        features: List of feature names
        target: Target column name
        model_builder: Function that returns untrained model
        n_splits: Number of validation folds
    
    Returns:
        results: Dict with per-fold metrics
        all_predictions: Combined predictions across all folds
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = {
        'fold': [],
        'train_size': [],
        'test_size': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc_roc': []
    }
    
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    
    X = df[features]
    y = df[target]
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # Split data chronologically
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train fresh model for this fold
        model = model_builder()
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred  # Fallback for models without probability
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # AUC-ROC (only if we have probability predictions)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = np.nan
        
        # Store results
        results['fold'].append(fold + 1)
        results['train_size'].append(len(train_idx))
        results['test_size'].append(len(test_idx))
        results['accuracy'].append(acc)
        results['precision'].append(prec)
        results['recall'].append(rec)
        results['f1'].append(f1)
        results['auc_roc'].append(auc)
        
        # Accumulate predictions
        all_y_true.extend(y_test.values)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
    
    results_df = pd.DataFrame(results)
    
    predictions = {
        'y_true': np.array(all_y_true),
        'y_pred': np.array(all_y_pred),
        'y_proba': np.array(all_y_proba)
    }
    
    return results_df, predictions


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Create annotated confusion matrix visualization.
    
    Confusion Matrix:
    
                  Predicted
                  0       1
    Actual  0   TN      FP
            1   FN      TP
    
    Where:
    - True Negative (TN): Correctly predicted low-risk
    - False Positive (FP): Predicted high-risk, was low-risk (Type I error)
    - False Negative (FN): Predicted low-risk, was high-risk (Type II error)
    - True Positive (TP): Correctly predicted high-risk
    
    Derived Metrics:
    
    Precision = TP / (TP + FP)    # When we predict high-risk, how often are we correct?
    Recall = TP / (TP + FN)       # Of all high-risk days, how many did we catch?
    F1 = 2 · (Precision · Recall) / (Precision + Recall)  # Harmonic mean
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Create annotated heatmap
    labels = ['Low Risk (0)', 'High Risk (1)']
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=500,
        height=500
    )
    
    return fig


def plot_roc_curve(y_true, y_proba, title="ROC Curve"):
    """
    Plot Receiver Operating Characteristic curve.
    
    Mathematical Definition:
    
    For each threshold θ ∈ [0, 1]:
    
    TPR(θ) = TP(θ) / (TP(θ) + FN(θ))    # True Positive Rate (Sensitivity)
    FPR(θ) = FP(θ) / (FP(θ) + TN(θ))    # False Positive Rate
    
    ROC Curve: Plot of (FPR, TPR) as θ varies from 0 to 1
    
    AUC-ROC: Area Under the Curve
    - Perfect classifier: AUC = 1.0
    - Random classifier: AUC = 0.5
    - Interpretation: Probability that model ranks random positive higher than random negative
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    fig = go.Figure()
    
    # Plot ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, 
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    # Plot diagonal (random classifier baseline)
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600,
        height=600,
        showlegend=True
    )
    
    return fig


def compare_models(results_dict):
    """
    Create comparative visualization of multiple models.
    
    Args:
        results_dict: Dict mapping model_name -> results_df from walk_forward_validation
    
    Returns:
        Plotly figure comparing F1 scores across folds
    """
    fig = go.Figure()
    
    for model_name, results_df in results_dict.items():
        fig.add_trace(go.Scatter(
            x=results_df['fold'],
            y=results_df['f1'],
            mode='lines+markers',
            name=model_name,
            line=dict(width=2),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title='Model Comparison: F1 Score Across Folds',
        xaxis_title='Fold',
        yaxis_title='F1 Score',
        width=800,
        height=500,
        showlegend=True
    )
    
    return fig


def get_classification_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate all classification metrics.
    
    Returns:
        Dict with comprehensive metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['auc_roc'] = np.nan
    
    # Add confusion matrix values
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negative'] = tn
        metrics['false_positive'] = fp
        metrics['false_negative'] = fn
        metrics['true_positive'] = tp
    
    return metrics


def feature_importance_plot(model, feature_names, top_n=10):
    """
    Plot feature importance for tree-based models.
    
    Feature Importance Interpretation:
    
    For Random Forest:
    Importance(X_j) = Average decrease in Gini impurity when splitting on feature X_j
    
    Higher value → feature is more discriminative for classification
    """
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig = go.Figure(go.Bar(
        x=[feature_names[i] for i in indices],
        y=importances[indices],
        marker_color='steelblue'
    ))
    
    fig.update_layout(
        title='Feature Importance (Random Forest)',
        xaxis_title='Feature',
        yaxis_title='Importance Score',
        width=800,
        height=500
    )
    
    return fig
