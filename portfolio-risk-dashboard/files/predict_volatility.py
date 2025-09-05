import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def predict_volatility_arima(returns, days_ahead=5):
    vol = returns.rolling(window=5).std().dropna()
    model = ARIMA(vol, order=(2,1,2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=days_ahead)
    return forecast
