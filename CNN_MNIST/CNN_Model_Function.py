#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import keras
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# In[19]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


# In[20]:


from keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
fig = plt.figure(1)
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(X_train[i], cmap = 'gray', interpolation = 'none')
    plt.title("Digit: {}".format(Y_train[i]))
    plt.xticks([])
    plt.yticks([])
    fig
#plt.show()


# In[21]:


#Reshape mnist dataset
print('before reshaping x_train', X_train.shape)
print('before reshaping x_test', X_test.shape)
#Then after reshaping:
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
print('x_train:', X_train.shape)
print('x_test:', X_test.shape)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
#Encoding Y to binary form
num_category = 10
Y_train = keras.utils.to_categorical(Y_train, num_category)
Y_test = keras.utils.to_categorical(Y_test, num_category)
print(Y_train.shape, Y_test.shape)


# In[22]:


def model(conv1_channel, conv1_ksize, conv2_channel, conv2_ksize, pool_size, drop_rate1, drop_rate2,FC_size, batch_size, epochs):
    #The conv_ksize is the dimension of kernel, for example  conv1_ksize = 3, kernel_size = (3,3)
    model = Sequential()
    #The initial conv layers (can be from 2-4)
    model.add(Conv2D(conv1_channel, kernel_size = (conv1_ksize, conv1_ksize), activation = 'relu', input_shape = input_shape))
    model.add(Conv2D(conv2_channel, kernel_size = (conv2_ksize, conv2_ksize), activation = 'relu'))
    '''
    if init_conv == 4:
        #model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
        #model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
    if init_conv == 3:
        #model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
    if init_conv == 2:
    '''
    model.add(MaxPooling2D(pool_size = (pool_size, pool_size)))
    model.add(Dropout(drop_rate1))
    model.add(Flatten())
    model.add(Dense(FC_size, activation = 'relu'))
    model.add(Dropout(drop_rate2))
    model.add(Dense(num_category, activation = 'softmax'))
    
    #compile the model
    model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])
    model_log = model.fit(X_train, Y_train, batch_size , epochs , verbose=1, validation_data=(X_test, Y_test))
    
    score = model.evaluate(X_test, Y_test, verbose=0)
    return score, model_log


# In[23]:


#using fixed parameters to test
conv1_channel = 32
conv1_ksize = 3
conv2_channel = 64
conv2_ksize = 3
pool_size = 2
drop_rate1 = 0.25
drop_rate2 = 0.5
FC_size = 128
batch_size = 128
epochs = 5

score, model_log = model(conv1_channel, conv1_ksize, conv2_channel, conv2_ksize, pool_size, drop_rate1, drop_rate2,FC_size, batch_size, epochs)


# In[24]:


print('Test Loss:', score[0])
print('Test accuracy:', score[1])

#accuracy plot
fig = plt.figure(2)
plt.subplot(2, 1, 1)
plt.axis([0,epochs,0.9,1])
plt.plot(model_log.history['acc'])
plt.plot(model_log.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc = 'lower right')
plt.yticks(np.arange(0.9,1,0.01))
plt.xticks(np.arange(0, epochs, 1))
#loss plot
fig = plt.figure(2)
plt.subplot(2, 1, 2)
plt.axis([0,epochs,0,0.3])
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc = 'upper right')
plt.yticks(np.arange(0,0.3,0.05))
plt.xticks(np.arange(0, epochs, 1))
plt.show()
# In[ ]:




