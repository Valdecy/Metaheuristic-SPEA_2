############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Strength Pareto Evolutionary Algorithm 2

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-SPEA_2, File: Python-MH-SPEA-2.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-SPEA_2>

############################################################################

# Required Libraries
import pandas as pd
import numpy  as np
import math
import matplotlib.pyplot as plt
import random
import os

# Function 1
def func_1():
    return

# Function 2
def func_2():
    return

# Function: Initialize Variables
def initial_population(population_size = 5, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2]):
    population = pd.DataFrame(np.zeros((population_size, len(min_values))))
    for i in range (0, len(list_of_functions)):
        name = str(i+1)
        name = "Fitness_" + name
        population[name] = 0.0
    for i in range(0, population_size):
        for j in range(0, len(min_values)):
             population.iloc[i,j] = random.uniform(min_values[j], max_values[j])      
        for k in range (1, len(list_of_functions) + 1):
            population.iloc[i,-k] = list_of_functions[-k](population.iloc[i,0:population.shape[1]-len(list_of_functions)])
    return population
    
# Function: Dominance
def dominance_function(solution_1, solution_2, number_of_functions = 2):
    count = 0
    dominance = True
    for k in range (1, number_of_functions + 1):
        if (solution_1.iloc[-k] <= solution_2.iloc[-k]):
            count = count + 1
    if (count == number_of_functions):
        dominance = True
    else:
        dominance = False       
    return dominance

# Function: Raw Fitness
def raw_fitness_function(population, number_of_functions = 2):    
    strength = pd.DataFrame(np.zeros((population.shape[0], 1)), columns = ['Strength'])
    raw_fitness = pd.DataFrame(np.zeros((population.shape[0], 1)), columns = ['Raw'])
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if(i != j):
                if dominance_function(solution_1 = population.iloc[i,:], solution_2 = population.iloc[j,:], number_of_functions = number_of_functions):
                    strength.iloc[i,0] = strength.iloc[i,0] + 1
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if(i != j):
                if dominance_function(solution_1 = population.iloc[i,:], solution_2 = population.iloc[j,:], number_of_functions = number_of_functions):
                    raw_fitness.iloc[j,0] = raw_fitness.iloc[j,0] + strength.iloc[i,0]
    return raw_fitness

# Function: Distance Calculations
def euclidean_distance(x, y):       
    distance = 0
    for j in range(0, len(x)):   
        distance = (x.iloc[j] - y.iloc[j])**2 + distance   
    return distance**(1/2)  

# Function: Fitness
def fitness_calculation(population, raw_fitness, number_of_functions = 2):
    k = int(len(population)**(1/2)) - 1
    fitness  = pd.DataFrame(np.zeros((population.shape[0], 1)), columns = ['Fitness'])
    distance = pd.DataFrame(np.zeros((population.shape[0], population.shape[0])))
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if(i != j):
                x = population.iloc[i, population.shape[1]-number_of_functions:].copy(deep = True)
                y = population.iloc[j, population.shape[1]-number_of_functions:].copy(deep = True)
                distance.iloc[i,j] =  euclidean_distance(x = x, y = y)                    
    for i in range(0, fitness.shape[0]):
        distance = distance.sort_values(by = i, axis = 1, ascending = True)
        fitness.iloc[i,0] = raw_fitness.iloc[i,0] + 1/(distance.iloc[i,k] + 2)
    return fitness

# Function: Sort Population by Fitness
def sort_population_by_fitness(population, fitness):
    idx = np.argsort(fitness['Fitness'].values)
    fitness_new = pd.DataFrame(np.zeros((population.shape[0], 1)), columns = ['Fitness'])
    population_new = pd.DataFrame(np.zeros((population.shape[0], population.shape[1])))  
    for i in range(0, population.shape[0]):
        fitness_new.iloc[i,0] = fitness.iloc[idx[i],0] 
        for k in range(0, population.shape[1]):
            population_new.iloc[i,k] = population.iloc[idx[i],k]
    return population_new, fitness_new

# Function: Selection
def roulette_wheel(fitness_new): 
    fitness = pd.DataFrame(np.zeros((fitness_new.shape[0], 1)))
    fitness['Probability'] = 0.0
    for i in range(0, fitness.shape[0]):
        fitness.iloc[i,0] = 1/(1+ fitness.iloc[i,0] + abs(fitness.iloc[:,0].min()))
    fit_sum = fitness.iloc[:,0].sum()
    fitness.iloc[0,1] = fitness.iloc[0,0]
    for i in range(1, fitness.shape[0]):
        fitness.iloc[i,1] = (fitness.iloc[i,0] + fitness.iloc[i-1,1])
    for i in range(0, fitness.shape[0]):
        fitness.iloc[i,1] = fitness.iloc[i,1]/fit_sum
    ix = 0
    random = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (random <= fitness.iloc[i, 1]):
          ix = i
          break
    return ix

# Function: Offspring
def breeding(population, fitness, min_values = [-5,-5], max_values = [5,5], mu = 1, list_of_functions = [func_1, func_2]):
    offspring = population.copy(deep = True)
    b_offspring = 0
    for i in range (0, offspring.shape[0]):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = random.sample(range(0, len(population) - 1), 1)[0]
        for j in range(0, offspring.shape[1] - len(list_of_functions)):
            rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            rand_b = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                
            if (rand <= 0.5):
                b_offspring = 2*(rand_b)
                b_offspring = b_offspring**(1/(mu + 1))
            elif (rand > 0.5):  
                b_offspring = 1/(2*(1 - rand_b))
                b_offspring = b_offspring**(1/(mu + 1))       
            offspring.iloc[i,j] = np.clip(((1 + b_offspring)*population.iloc[parent_1, j] + (1 - b_offspring)*population.iloc[parent_2, j])/2, min_values[j], max_values[j])           
            if(i < population.shape[0] - 1):   
                offspring.iloc[i+1,j] = np.clip(((1 - b_offspring)*population.iloc[parent_1, j] + (1 + b_offspring)*population.iloc[parent_2, j])/2, min_values[j], max_values[j]) 
        for k in range (1, len(list_of_functions) + 1):
            offspring.iloc[i,-k] = list_of_functions[-k](offspring.iloc[i,0:offspring.shape[1]-len(list_of_functions)])
    return offspring 

# Function: Mutation
def mutation(offspring, mutation_rate = 0.1, eta = 1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2]):
    d_mutation = 0            
    for i in range (0, offspring.shape[0]):
        for j in range(0, offspring.shape[1] - len(list_of_functions)):
            probability = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            if (probability < mutation_rate):
                rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                rand_d = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                     
                if (rand <= 0.5):
                    d_mutation = 2*(rand_d)
                    d_mutation = d_mutation**(1/(eta + 1)) - 1
                elif (rand > 0.5):  
                    d_mutation = 2*(1 - rand_d)
                    d_mutation = 1 - d_mutation**(1/(eta + 1))                
                offspring.iloc[i,j] = np.clip((offspring.iloc[i,j] + d_mutation), min_values[j], max_values[j])                        
        for k in range (1, len(list_of_functions) + 1):
            offspring.iloc[i,-k] = list_of_functions[-k](offspring.iloc[i,0:offspring.shape[1]-len(list_of_functions)])
    return offspring 

# SPEA-2 Function
def strength_pareto_evolutionary_algorithm_2(population_size = 5, archive_size = 5, mutation_rate = 0.1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2], generations = 50, mu = 1, eta = 1):        
    count = 0   
    population = initial_population(population_size = population_size, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions) 
    archive = initial_population(population_size = archive_size, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions)     
    while (count <= generations):       
        print("Generation = ", count)
        population = pd.concat([population, archive])
        raw_fitness   = raw_fitness_function(population, number_of_functions = len(list_of_functions))
        fitness    = fitness_calculation(population, raw_fitness, number_of_functions = len(list_of_functions))        
        population, fitness = sort_population_by_fitness(population, fitness)
        population, archive, fitness = population.iloc[0:population_size,:], population.iloc[0:archive_size,:], fitness.iloc[0:archive_size,:]
        population = breeding(population, fitness, mu = mu, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions)
        population = mutation(population, mutation_rate = mutation_rate, eta = eta, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions)             
        count = count + 1              
    return archive

######################## Part 1 - Usage ####################################

# Schaffer Function 1
def schaffer_f1(variables_values = [0]):
    y = variables_values[0]**2
    return y

# Schaffer Function 2
def schaffer_f2(variables_values = [0]):
    y = (variables_values[0]-2)**2
    return y

# Shaffer Pareto Front
schaffer = pd.DataFrame(np.arange(0.0, 2.0, 0.01))
schaffer['Function 1'] = 0.0
schaffer['Function 2'] = 0.0
for i in range (0, schaffer.shape[0]):
    schaffer.iloc[i,1] = schaffer_f1(variables_values = [schaffer.iloc[i,0]])
    schaffer.iloc[i,2] = schaffer_f2(variables_values = [schaffer.iloc[i,0]])

schaffer_1 = schaffer.iloc[:,1]
schaffer_2 = schaffer.iloc[:,2]

# Calling SPEA-2 Function
spea_2_schaffer = strength_pareto_evolutionary_algorithm_2(population_size = 50, archive_size = 50, mutation_rate = 0.1, min_values = [-5], max_values = [5], list_of_functions = [schaffer_f1, schaffer_f2], generations = 100, mu = 5, eta = 5)

# Graph Pareto Front Solutions
func_1_values = spea_2_schaffer.iloc[:,-2]
func_2_values = spea_2_schaffer.iloc[:,-1]
ax1 = plt.figure(figsize = (15,15)).add_subplot(111)
plt.xlabel('Function 1', fontsize = 12)
plt.ylabel('Function 2', fontsize = 12)
ax1.scatter(func_1_values, func_2_values, c = 'red',   s = 25, marker = 'o', label = 'SPEA-2')
ax1.scatter(schaffer_1,    schaffer_2,    c = 'black', s = 2,  marker = 's', label = 'Pareto Front')
plt.legend(loc = 'upper right')
plt.show()
