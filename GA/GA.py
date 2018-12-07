
# coding: utf-8

# In[16]:


import numpy

def cal_pop_fitness(equation_inputs, pop):

    #calculating the fitness value of each solution in the current population.
    #the fitness function calculates the sum of products of weights and inputs.
    
    fitness = numpy.sum(pop*equation_inputs, axis = 1)
    return fitness


# In[17]:


def select_mating_pool(pop, fitness, num_parents):
    #select the best individuals as parents 
    parents = numpy.empty((num_parents, pop.shape[1]))
    
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -9999999999
    return parents

