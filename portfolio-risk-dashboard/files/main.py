import pandas as pd
import matplotlib.pyplot as plt
from metrics import cumulative_returns

df = pd.read_csv("stock_data.csv", index_col=0, parse_dates=True)
cum_ret = cumulative_returns(df)
cum_ret.plot(figsize=(10, 5))
plt.title("Cumulative Returns")
plt.show()
