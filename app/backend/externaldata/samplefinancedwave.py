import yfinance as yf
import numpy as np
import json
import pandas as pd
import dimod
from itertools import product
from dimod import Integer, Binary, quicksum, ConstrainedQuadraticModel




Data = yf.download(['MSFT','TSLA','AAPL'], start ='2022-01-01')['Adj Close']
print(Data.head())
returns = Data.pct_change(1).dropna()
cov = np.array(returns.cov())
mean = returns.mean()
prices = Data.iloc[-1]


iv = [Integer(f'x_{i}', upper_bound=10) for i in range(3)]

cqm = ConstrainedQuadraticModel()
cqm.set_objective(quicksum(quicksum(iv[i]*iv[j]*prices[i]*prices[j]*cov[i,j] for j in range(3)) for i in range(3)))
budget = 1000
cqm.add_constraint(quicksum(prices[i]*iv[i] for i in range(3)) <=budget)
cqm.add_constraint(quicksum(prices[i]*iv[i] for i in range(3))  >= 0.98 * budget)

bqm,invert = dimod.cqm_to_bqm(cqm)

qubo1 = bqm.to_numpy_vectors()

Linear_Biases = qubo1.linear_biases
n = len(qubo1.linear_biases)

Row_Indices = qubo1.quadratic.row_indices
Col_Indices = qubo1.quadratic.col_indices
Biases = qubo1.quadratic.biases

matrix = np.zeros((n,n))
np.fill_diagonal(matrix, Linear_Biases)

# add quadratic biases to the corresponding positions
for r, c, b in zip(Row_Indices, Col_Indices, Biases):
    matrix[r][c] += b
    if r != c:
        # ensure the matrix remains symmetric
        matrix[c][r] += b

# Convert it to a DataFrame
df = pd.DataFrame(matrix)

print (df.shape) 

df.to_excel('qubo31_08_2023.xlsx', sheet_name='Sheet1', index=False)