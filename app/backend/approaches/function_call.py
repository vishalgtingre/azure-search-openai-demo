import json
from datetime import datetime


# AVAILABLE_FUNCTIONS = {
#     "ask_user_investment_appetite":ask_user_investment_appetite
# }

TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "ask_user_investment_appetite",
                "description": "Help user for investing money and ask basic details reagrding his investment appetite.Invest money, maximise return, minimise risk, create portfolio, make me richer, Portfolio optimization, Portfolio optimisation, investment advice, investment advise",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_stock_data",
                "description": "Get stock data from yahoo finance and distribute stock based on budget, list of stocks and duration (example, i want to invest 1000 in apple, tesla for 1 year)",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
        },
]

FUNCTION_SCHEMA = [
        {
            "name": "invest_money",
            "description": "Invest money based on user investment appetite . (example, i want to invest 1000 in apple, tesla for 1 year)",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "ask_user_investment_appetite",
            "description": "Help user for investing money and ask basic details reagrding his investment appetite.Invest money, maximise return, minimise risk, create portfolio, make me richer, Portfolio optimization, Portfolio optimisation, investment advice, investment advise",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
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
    '''Ask basic user investment appetite related questions'''
    print("--------get_stock_distribution--", args, kwargs)
    return json.dumps({"apple": 50, "tesla": 30, "google": 20})