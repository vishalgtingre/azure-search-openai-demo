
import yfinance as yf
import numpy as np
from itertools import product
import json
import random
import numpy as np
import pandas as pd
import sys
import yfinance as yf
import pandas as pd

tickers = ['MSFT', 'AMZN', 'AAPL', 'NFLX', 'GOOG']
start_date = '2022-01-01'
end_date = '2022-12-31'

extData = pd.DataFrame()
for ticker in tickers:
    temp_data = yf.download(ticker, start=start_date, end=end_date)
    temp_data['Ticker'] = ticker
    extData = pd.concat([extData, temp_data])

extData = extData.reset_index()
# Print the results
meanReturn = '' 
covMatrix = ''
print(extData)
print(meanReturn)
print(covMatrix)
