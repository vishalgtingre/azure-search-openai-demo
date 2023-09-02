import yfinance as yf

# Define the ticker symbols for the stocks
tickers = 1

# Download the OHLC data for the stocks
data = yf.download(tickers, start='2022-01-01', end='2022-12-31')

# Select the 'Adj Close' prices
prices = data2

# Print the first 5 rows of the data
print(prices.head())