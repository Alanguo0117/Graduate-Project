#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from CNN_Model_Function import model


# In[3]:


def pop_gen(pop_size):
    #pop_size = 2
    conv1_channel = np.random.randint(low = 2, high = 65, size = pop_size) 
    conv1_ksize = np.random.randint(low = 2, high = 11, size = pop_size)
    conv2_channel = np.random.randint(low = 16, high = 129, size = pop_size)
    conv2_ksize = np.random.randint(low = 2, high = 11, size = pop_size)
    pool_size = np.random.randint(low = 2, high = 11, size = pop_size)
    drop_rate1 = np.random.uniform(low = 0.1, high = 0.9, size = pop_size)
    drop_rate2 = np.random.uniform(low = 0.1, high = 0.9, size = pop_size)
    FC_size = np.random.randint(low = 16, high = 257, size = pop_size)
    #batch_size = np.random.randint(low = 1, high = 257, size = pop_size)
    
    pop_matrix = np.array([conv1_channel,conv1_ksize,conv2_channel,conv2_ksize,                         pool_size,FC_size,drop_rate1,drop_rate2])
    pop_matrix = pop_matrix.T
    #print(pop_matrix)
    #print(pop_matrix.shape)
    return pop_matrix


# In[8]:


#pop_matrix = offspring_crossover
#print(pop_matrix)


# In[ ]:


def cal_fitness(pop_matrix, pop_size):
    fitness = np.zeros(pop_size)
    #test_equ = np.random.randn(8)
    score = np.zeros((pop_size, 2))
    for i in range(pop_size):
        gafit = model(pop_matrix[i,0],pop_matrix[i,1],pop_matrix[i,2],pop_matrix[i,3],pop_matrix[i,4],pop_matrix[i,5],pop_matrix[i,6],pop_matrix[i,7])
        fitness[i] = gafit[0]
        score[i,:] = gafit[1]
    #print(fitness)
    return fitness, score


# In[5]:


def select_parents(pop_matrix, fitness, num_parents):
    #fitness = [5,2,3,6]
    #pop_matrix = np.array([[1,1,2,2,4,6,3,5],[2,2,3,3,4,6,7,8],[3,3,4,4,8,9,6,3],[4,4,5,5,2,2,3,4]])
    #num_parents = 2
    parents = np.empty((num_parents, pop_matrix.shape[1]))
    for i in range(num_parents):
        max_idx = np.where(fitness == np.max(fitness))
        #print(max_idx)
        max_idx = max_idx[0][0]
        #print(max_idx)
        parents[i, :] = pop_matrix[max_idx, :]
        fitness[max_idx] = -99999999999
    return parents
    #print(parents)


# In[6]:


def crossover(parents, offspring_size):
    #offspring_size = (2,8)
    offspring = np.empty(offspring_size)
    crossover_pt = np.uint8(offspring_size[1]/2)
    for k in range(offspring_size[0]):
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]
        offspring[k, 0:crossover_pt] = parents[parent1_idx, 0:crossover_pt]
        offspring[k, crossover_pt:] = parents[parent2_idx, crossover_pt:]
    return offspring
    #print(offspring)


# In[7]:


def mutation(offspring_crossover):
    #offspring_crossover = offspring
    #random_idx = np.random.randint(low = 0, high = 5, size = 1)
    #print(random_idx)
    for j in range(offspring_crossover.shape[0]):
        random_idx = np.random.randint(low = 0, high = 6, size = 1)
        random_value = np.random.randint(low = -1, high = 1 , size=1)
        offspring_crossover[j, random_idx] = offspring_crossover[j, random_idx] + random_value
    return offspring_crossover
    #print(offspring_crossover)


# In[ ]:





# In[ ]:




