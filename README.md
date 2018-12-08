# Graduate-Project
This is for the preparation for AM graduate.  
Using genetic algorithm to find optimal CNN architecture.   
## About GA  

For now, I did a simple example with genetic algorithm finding a maximum value of a linear function "y=w1x1+w2x2+w3x3+w4x4+w5x5+w6x6". Given a set of X=[x1,x2,x3,x4,x5,x6], find the parameter wi can make the value of function go maximum.   
We set different W=[w1,w2,w3,w4,w5,w6] as the population, and assuming an individaul in the set has only one chromosome with 6 genes. After crossover and mutation for five generations the fitness function goes bigger and bigger. It covergence to a certain value. This shows the method working good.  
From this I found it can be possible to use GA as a optimizer for some hyperparameter problem. GA has the following advantage:  
1. GA actually starts at searching all over the population, not start at any single point, we can aviod missing some critical points.  
2. GA allows us search optimal solution when the number of parameters is very large.  
3. We do not need the objective function convex or concave, using GA we do not need to find gradients or some derivative methods.    
4. GA is good at the problem with multi local optima.
### The code of this example is in GA folder named as "population.ipynb", other files are just reference PDFs and some testing or abandoned code.  

## About MNIST
