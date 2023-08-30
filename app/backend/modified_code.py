
import pandas as pd
import yfinance as yf

# Define the list of assets and the budget
assets = ['INTU', 'ISRG', 'HAS']
budget = 1000

# Get the OHLC data for the assets
extData = yf.download(assets, start='2022-01-01', end='2022-01-31')

# Calculate the mean returns and covariance matrix
returns = pd.DataFrame.pct_change(extData['Adj Close'])
meanReturn = returns.mean()
covMatrix = returns.cov()
