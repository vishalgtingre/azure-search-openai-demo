import json
from datetime import datetime

#for portfolio optimization
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

from approaches.vehicle_routing import calculate_optimize_vehicle_route
from approaches.vehicle_routing_dwave import calculate_optimize_vehicle_route_dwave

import requests

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
                "description": "Get stock data from yahoo finance and distribute stock based on budget, list of stocks and duration (example, i want to invest 1000 in apple, tesla for 1 year) This will return the result based on Dwave's Quantum Annealer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_list": {
                            "type": "array",
                            "description": "give ticker symbols for list of the stocks",
                            "items": {
                                "type": "string",
                            },
                        },
                        "budget": {
                            "type": "integer",
                            "description": " Budget of the user"
                        }
                        },
                    "required": ["search_list", "budget"]
                     }
                }
        },
        {
            "type": "function",
            "function": {
                "name": "ask_vehicle_route_details",
                "description": "Assist logistics, fleet, or transportation managers in optimizing their vehicle routing and scheduling. Vehicle routing problem optimization, fleet management.  Assit in streamline logistics operations, enhance route efficiency, reduce transportation costs, and improve overall fleet management. Key features include route optimization, efficient resource allocation, and strategic planning for vehicle deployment. Ideal for managing deliveries, optimizing travel routes, and ensuring timely operations in logistics and transportation management.",
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
                "name": "optimize_vehicle_route",
                "description": "Get optimized vehicle route based on fleet size and number of locations. (example, i have 2 vehicles and 5 locations for vehicle routing) This will return the result based on IBM qiskit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "number_of_vehicles": {
                            "type": "integer",
                            "description": "give number of vehicles.",
                        },
                        "number_of_nodes": {
                            "type": "integer",
                            "description": " Number of nodes/locations."
                        },
                        "list_of_cities": {
                            "type": "array",
                            "description": "list of the cities",
                            "items": {
                                "type": "string",
                            },
                        }
                        },
                    "required": ["number_of_vehicles", "list_of_cities"]
                }
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

def get_stock_data(*args, **kwargs):
    '''Ask  user to provide stocks and budget for investment '''
    parameters = args[0]
    input_data = parameters.get("search_list")

    if isinstance(parameters.get("search_list"), str):
        input_data = parameters.get("search_list").split(',')

    nInpCnt = len(input_data)
    budget = parameters.get('budget')

    Data = yf.download(input_data, start ='2022-01-01')['Adj Close']
    
    assets = Data.columns.tolist()
    returns = Data.pct_change(1).dropna()
    cov = np.array(returns.cov())
    mean = returns.mean()
    prices = Data.iloc[-1]
    iv = [Integer(f'x_{i}', upper_bound=10) for i in range(nInpCnt)]
    cqm = ConstrainedQuadraticModel()
    cqm.set_objective(quicksum(quicksum(iv[i]*iv[j]*prices[i]*prices[j]*cov[i,j] for j in range(nInpCnt)) for i in range(nInpCnt)))
    cqm.add_constraint(quicksum(prices[i]*iv[i] for i in range(nInpCnt)) <=budget)
    cqm.add_constraint(quicksum(prices[i]*iv[i] for i in range(nInpCnt))  >= 0.98 * budget)
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
        final_output[assets[idx]] =  {"allocation_percentage" : optim_weights[idx], "investment_amount" : optim_weights[idx] * budget}
    
    result = {"data": final_output, "user note": "This is real time calculation using ConstrainedQuadraticModel.", "formatting" : "show data in tabular format.", "chatgpt instruction": "DO NOT add any disclaimer by yourself."}
    return json.dumps(result)


def ask_vehicle_route_details(*args, **kwargs):
    '''Ask vehicle route related questions'''
    return "To help you optimize vehicle route please provide number of vehciles and list of citites"


def optimize_vehicle_route(*args, **kwargs):
    '''Ask vehicle route related questions'''
    parameters = args[0]

    number_of_nodes = len(parameters.get("list_of_cities"))
    number_of_vehicles = parameters.get("number_of_vehicles")
    #fetch city co-ordinates, and set vectorOfVolume to 100 by default 
    vectorOfVolume = []
    vectorOfCapacity =[]
    city_coordinates = []
    for city in parameters.get("list_of_cities"):
        try:
            vectorOfVolume.append(100)
            url = 'https://nominatim.openstreetmap.org/search?addressdetails=1&q='+city+'&format=jsonv2&limit=1'
            response = requests.get(url)
            if response and response.status_code == 200:
                data = response.json()
                location = data[0]
                city_coordinates.append([float(location['lat']), float(location['lon'])])
        except Exception as e:
            city_coordinates.append([random.randint(0, 100), random.randint(0, 100)])

    for i in range(0, number_of_vehicles):
        vectorOfCapacity.append(int(100 * number_of_nodes /number_of_vehicles ) + 50) 
        
    if number_of_nodes <=3:
        vehicle_path, cost = calculate_optimize_vehicle_route(number_of_nodes, number_of_vehicles)
        vehicle_path = vehicle_path.tolist()
        return json.dumps({"quantum cost": cost, "vehicle_path":vehicle_path})
    else:
        data = calculate_optimize_vehicle_route_dwave(number_of_vehicles, number_of_nodes, vectorOfVolume, vectorOfCapacity, city_coordinates)
        return json.dumps(data)

