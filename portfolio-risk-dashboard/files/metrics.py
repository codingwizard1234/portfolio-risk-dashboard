import numpy as np
import pandas as pd

def cumulative_returns(price_df, weights=None):
    daily_ret = price_df.pct_change().dropna()
    if weights is not None:
        port_ret = daily_ret.dot(weights)
        return (1 + port_ret).cumprod()
    else:
        return (1 + daily_ret).cumprod()

def volatility(price_df, weights):
    daily_ret = price_df.pct_change().dropna()
    port_ret = daily_ret.dot(weights)
    return port_ret.std() * np.sqrt(252)

def sharpe_ratio(price_df, weights, risk_free_rate=0.0):
    daily_ret = price_df.pct_change().dropna()
    port_ret = daily_ret.dot(weights)
    excess_ret = port_ret - risk_free_rate/252
    return np.sqrt(252) * excess_ret.mean() / excess_ret.std()

def max_drawdown(price_df, weights):
    daily_ret = price_df.pct_change().dropna()
    port_ret = daily_ret.dot(weights)
    cum_ret = (1 + port_ret).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    return drawdown.min()

def value_at_risk(price_df, weights, alpha=0.05):
    daily_ret = price_df.pct_change().dropna()
    port_ret = daily_ret.dot(weights)
    return np.percentile(port_ret, 100 * alpha)

def portfolio_return(price_df, weights):
    daily_ret = price_df.pct_change().dropna()
    return daily_ret.dot(weights)
