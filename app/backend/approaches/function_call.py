import json
from datetime import datetime
import yfinance as yf
import numpy as np
from itertools import product
import random 
import pandas as pd
from pandas_datareader.data import DataReader
from dimod import Integer, Binary
from dimod import quicksum 
from dimod import ConstrainedQuadraticModel
from dwave.system import LeapHybridDQMSampler, LeapHybridCQMSampler 


TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "ask_user_investment_appetite",
                "description": "Help user for investing money and ask basic details reagrding his investment appetite.Invest money, maximise return, minimise risk, create portfolio, make me richer, Portfolio optimization, Portfolio optimisation, investment advice, investment advise",
                "parameters": {
                    "type": "object",
                    "properties": {
                        },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_stock_data",
                "description": "Get stock data from yahoo finance and distribute stock based on budget, list of stocks and duration (example, i want to invest 1000 in apple, tesla for 1 year)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_list": {
                            "type": "string",
                            "description": " list of the stocks "
                        },
                        "budget": {
                            "type": "string",
                            "description": " Budget of the user"
                        }
                            }
                        },
                    "required": ["search_list","budget"]
                }
        }
]

def ask_user_investment_appetite(*args, **kwargs):
    '''Ask basic user investment appetite related questions'''

    return "To help you invest please provide budget, list of stocks and duration"

def get_current_time(location="", *args, **kwargs):
    if not location:
        return "Please provide current location."
    now = datetime.now(location)
    current_time = now.strftime("%I:%M:%S %p")
    return current_time

def get_stock_data(search_list="",budget="",*args, **kwargs):
    '''Ask  user to provide stocks and budget for investment '''
    
    print(search_list)
    print("--------get_stock_distribution--")
    # return json.dumps({"apple": 50, "tesla": 30, "google": 20})
    print(budget)
    
    # Convert the string to a dictionary
    input_dict = json.loads(search_list)

    # Extract the budget and convert it to an integer
    budget = int(input_dict["budget"])

    # Extract the search list and convert it to a list of strings
    

# Example of processing the data
    print("Budget:", budget)
    print("Companies to invest in:")
    for company in search_list:
        print(f"- {company}")
    
    ticker_symbols = input_dict["search_list"].split(", ")
    
    Data = yf.download(ticker_symbols, start ='2022-01-01')['Adj Close']
    print(Data)
    assets = Data.columns.tolist()
    returns = Data.pct_change(1).dropna()
    cov = np.array(returns.cov())
    mean = returns.mean()
    prices = Data.iloc[-1]
    iv = [Integer(f'x_{i}', upper_bound=10) for i in range(4)]
    cqm = ConstrainedQuadraticModel()
    cqm.set_objective(quicksum(quicksum(iv[i]*iv[j]*prices[i]*prices[j]*cov[i,j] for j in range(4)) for i in range(4)))
    budget = 10000
    cqm.add_constraint(quicksum(prices[i]*iv[i] for i in range(4)) <=budget)
    cqm.add_constraint(quicksum(prices[i]*iv[i] for i in range(4))  >= 0.98 * budget)
    cqm_sampler = LeapHybridCQMSampler(token='DEV-c398268cb2d92fe3038d906bd2bfb8b4dba9d923')
    sample_set = cqm_sampler.sample_cqm(cqm,label='investment_with_4 optimization new')

    a = sample_set.aggregate().record
    cols  = list(a.dtype.names)
    b = []
    for i in range(a.shape[0]):
        b.append(list (a[i]))
        result = pd.DataFrame(data=b,columns=cols)
        result = result.sort_values('energy', ascending= False)
    
    #index_names = result[result['is_feasible'] == False ].index
    #result.drop(index_names,inplace=True)
    ans = result['sample'].iloc[-1]
    optim_weights = ans 
    number_stocks = np.array(list(optim_weights))
    number_stocks = np.reshape(number_stocks,(number_stocks.shape[0],))
    prices = np.reshape(prices,(prices.shape[0],))
    optim_weights = np.multiply(number_stocks,prices)
    optim_weights = optim_weights / np.sum(optim_weights)
    final_output = {}
    for idx in range(0, len(assets)):
        final_output[assets[idx]] = optim_weights[idx]
    return json.dumps(final_output)
