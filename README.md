# Graduate-Project
This is for the preparation for AM graduate.  
Using genetic algorithm(GA) to find optimal convolution neural network(CNN) architecture.   
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
Actually I get into machine learing with MNIST problem.  
The data structure of MNIST is very typical. We transform the hand-written figures of number 0 to 9 writed by human into matrices with entry equals 0 or 1, which means the black and white area of the figures.  

In the folder ML I tried using API from tensorflow(file named "mnist_gen.ipynb") and sklearn(file named "Untitled.ipynb") to do regression for MNIST problem.  

## The next step
Getting into using CNN to solve MNIST first and then try to use GA to do hyperparameter tuning.

## Using CNN to solve MNIST
After learning CNN algorithm, I implement in MNIST problem. In this example, I am using an CNN architecture with 2 Conv layers, the 1st filter is 3x3x32 and the second is 3x3x64 (32 and 64 channels). This two Conv layers is followed by a max pooling layer(2x2). Then I use a dropout node to accelerate convergence. Finally flatten the 3-D tensor and use 2 fully connected  layer to generate the out put. In the CNN all activate function except the last layer is Relu function and the last layer is using Softmax function for a multi classification.  
This CNN is mainly using Keras package and using matplotlib to generate plots. The accuracy and convergence rate is good.  
The hyperparameters in this problem is mainly the dimension and numbers of channels of filters, and we can also take the dropout rate as a hyperparameter. I am using (3x3x32), (3x3x64), and a 2x2 max pooling layer according some well-built neural network by some data scientist. The purpose of my project is using GA to tune CNN architecture. So for the next step, we am going to focus on how to build a effective fitness function for GA. I am considering to use a fitness function in the form of:
###                     fitness = A x training_accuracy + B x test_accuracy - C x training_loss - D x test_loss 
A,B,C,D are all positive numbers, and they are taken as the hyperparamter of GA for tuning. Then we assume individuals in this scenario have one chromosome s.t. Chro[fliter1_dimension, filter2_dimension, poolinglayer_dimension, dropout_rate], so this chromosome literally has 9(3+3+1+2) genes. Chro is generate randomly in a given interval to make sure convergence. I'll see what can I do in this week.
