from data_fetch import fetch_data
from metrics import portfolio_return
def scenario_performance(tickers,weights,start,end):
    df=fetch_data(tickers,start,end)
    return portfolio_return(df,weights).cumsum()[-1]