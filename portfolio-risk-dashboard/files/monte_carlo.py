import numpy as np
import pandas as pd
from metrics import portfolio_return
def monte_carlo_simulation(price_df,weights,num_simulations=5000,days=252):
    port_daily_ret = portfolio_return(price_df, weights)
    mean_ret = port_daily_ret.mean()
    vol = port_daily_ret.std()
    
    results = []
    for _ in range(num_simulations):
        simulated_prices = np.random.normal(mean_ret, vol, days)
        results.append(simulated_prices.sum())
        
    return pd.Series(results)