#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import GA


# In[3]:


#The paratmeter for GA driver
pop_size = 4
num_parents = 2
num_gen = 2


# In[4]:


#generate initial population
pop_matrix = GA.pop_gen(pop_size)
#iteration for GA
for i in range(num_gen):
    print("Generation :", i)
    #calculate fitness
    fitness, score = GA.cal_fitness(pop_matrix, pop_size)
    #output results to txt file
    max_idx = np.where(fitness == np.max(fitness))
    best_individual = pop_matrix[max_idx[0][0],:]
    best_score = score[max_idx[0][0],:]
    print(best_score)
    f = open('best_individual.txt','ab')
    np.savetxt(f,best_individual)
    f.close()
    f = open('pop_scores.txt','ab')
    np.savetxt(f,score)
    f.close()
    #select parents
    parents = GA.select_parents(pop_matrix, fitness, num_parents)
    #crossover 
    offspring_crossover = GA.crossover(parents, (pop_size - parents.shape[0],pop_matrix.shape[1]))
    #mutation
    offspring_mutation = GA.mutation(offspring_crossover)
    #getting new population
    pop_matrix[0:parents.shape[0], :] = parents
    pop_matrix[parents.shape[0]:, :]=offspring_mutation


# In[ ]:




