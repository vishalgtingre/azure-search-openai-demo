from dimod import ConstrainedQuadraticModel, CQM, SampleSet
from dimod import Binary, quicksum
from dwave.system import LeapHybridCQMSampler
from dwave.cloud.client import Client
from dwave.cloud import config
import numpy as np
import pandas as pd
import ast
import itertools

import random
import math

import time
import re

import random


def calculate_optimize_vehicle_route_dwave(nbOfVehicle = 5, nbOfPointToCluster = 10, vectorOfVolume = [], vectorOfCapacity =[], city_coordinates =[], city_list = []):
    cqm = ConstrainedQuadraticModel()
    # Define the binary variables
    x = {(i, d): Binary('x{}_{}'.format(i, d)) for i in range(nbOfPointToCluster) for d in range(nbOfVehicle)}

    # Generate random costs between each pair of cities
    matrixOfCost = [[0 if i == j else math.sqrt((city_coordinates[i][0] - city_coordinates[j][0])**2 + 
                                             (city_coordinates[i][1] - city_coordinates[j][1])**2) 
                 for j in range(nbOfPointToCluster)] for i in range(nbOfPointToCluster)]
    print("---matrixOfCost--", matrixOfCost, "------city_coordinates", city_coordinates)
    
    # Define the objective function
    objective = quicksum(matrixOfCost[i][j] * x[(i,d)] * x[(j,d)] for i in range(nbOfPointToCluster) for j in range(i+1, nbOfPointToCluster) for d in range(nbOfVehicle))
    cqm.set_objective(objective)

    # Define the constraints
    for d in range(nbOfVehicle):
        cqm.add_constraint(x[(0,d)] == 1)
    for d in range(nbOfVehicle):
        cqm.add_constraint(quicksum(vectorOfVolume[i] * x[(i,d)] for i in range(nbOfPointToCluster)) <= vectorOfCapacity[d])
    for i in range(1,nbOfPointToCluster):
        cqm.add_constraint(quicksum(x[(i,d)] for d in range(nbOfVehicle)) == 1)


    #We get our solution
    cqm_sampler = LeapHybridCQMSampler(token='DEV-c398268cb2d92fe3038d906bd2bfb8b4dba9d923')
    sample_set = cqm_sampler.sample_cqm(cqm,label='clustering_Sample')



    # Assuming sample_set is the SampleSet object from D-Wave
    a = sample_set.aggregate().record
    cols = list(a.dtype.names)
    b = []

    for i in range(a.shape[0]):
        b.append(list(a[i]))
    result = pd.DataFrame(data=b, columns=cols)
    result = result.sort_values('energy', ascending=False)


    index_names = result[result['is_feasible'] == False].index
    result.drop(index_names, inplace=True)

    # result.to_csv('C:/Users/amogh/Downloads/learning-20231104T155542Z-001/learning/cvrp/clustering_new3.csv')
    ans = result['sample'].iloc[-1]
    print(ans)

    # Assuming ans is a numpy array and city_coordinates, nbOfVehicle are defined
    # Reshape ans to match the number of cities and number of vehicles (clusters)
    cities_clusters = ans.reshape((-1, nbOfVehicle))


    # Calculate and display the total distance for each vehicle
    total_distances = np.zeros(nbOfVehicle)
    for d in range(nbOfVehicle):
        last_city = 0  # Assuming the route for each vehicle starts at city 0 (the depot)
        for city in range(1, len(city_coordinates)):
            if cities_clusters[city, d] == 1:
                total_distances[d] += matrixOfCost[last_city][city]
                last_city = city
        # Add distance back to the depot
        total_distances[d] += matrixOfCost[last_city][0]
        print(f"Total distance for Vehicle {d+1}: {total_distances[d]}")


    # Initialize a list to hold the information
    vehicles_info = []
    # Calculate the information for each vehicle
    for d in range(nbOfVehicle):
        # Extract the route for vehicle 'd'
        route = [i for i in range(nbOfPointToCluster) if cities_clusters[i, d] == 1]
        # Calculate the number of cities
        num_cities_covered = len(route)
        # Calculate the total distance traveled
        total_distance = sum(matrixOfCost[route[i]][route[i + 1]] for i in range(len(route) - 1))
        # Add the round trip to the depot if applicable
        total_distance += matrixOfCost[route[-1]][0] + matrixOfCost[0][route[0]]
        # Append the information to the list
        vehicles_info.append({
            'Vehicle': f'Vehicle {d + 1}',
            'Number of cities covered': num_cities_covered,
            'Total Distance Travelled': total_distance
        })
    # Convert the list to a DataFrame
    vehicles_df = pd.DataFrame(vehicles_info)
    # Optionally, you can round the total distance values to 3 decimal places
    vehicles_df['Total Distance Travelled'] = vehicles_df['Total Distance Travelled'].round(3)
    # Display the DataFrame
    print(vehicles_df)

    # Initialize a list to hold the information
    vehicles_info = []
    # Calculate the information for each vehicle
    for d in range(nbOfVehicle):
        route_city_list = []
        # Extract the route for vehicle 'd'
        route = [i for i in range(nbOfPointToCluster) if cities_clusters[i, d] == 1]  
        for city in route:
            route_city_list.append(city_list[city])
        # Calculate the total demand for the route
        total_demand_of_cluster = sum(vectorOfVolume[city] for city in route)
        # Calculate the capacity utilization
        capacity_utilization = total_demand_of_cluster / vectorOfCapacity[d]
        # Calculate the number of cities covered
        num_cities_covered = len(route)
        # Calculate the total distance traveled
        total_distance = sum(matrixOfCost[route[i]][route[i + 1]] for i in range(len(route) - 1))
        total_distance += matrixOfCost[route[-1]][0] + matrixOfCost[0][route[0]]  # Complete the round trip
        # Append the information to the list
        print("--------total", total_distance)
        vehicles_info.append({
            'Cluster': f'Cluster {d + 1}',
            'Total demand of the cluster': total_demand_of_cluster,  # Include the total demand
            'Vehicle': f'Vehicle {d + 1}',
            'Vehicle Capacity': vectorOfCapacity[d],
            'Capacity utilization': f"{capacity_utilization:.3f}",
            'Number of cities covered': num_cities_covered,
            'Total Distance Travelled': f"{total_distance:.3f}",
            "Route":route_city_list
        })
    print("----vehicles_info-", vehicles_info)
    return vehicles_info

