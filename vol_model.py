# vol_model.py - GARCH model for volatility forecasting
# Uses GARCH(1,1) to forecast one-step ahead conditional volatility

from arch import arch_model

def garch_forecast(returns):
    # Fit GARCH(1,1) model on returns scaled to basis points for numerical stability
    model = arch_model(returns * 100, vol="Garch", p=1, q=1)
    res = model.fit(disp="off")
    
    # Generate one-step ahead variance forecast
    forecast = res.forecast(horizon=1)
    
    # Extract and return the forecasted conditional variance
    return forecast.variance.values[-1][0]