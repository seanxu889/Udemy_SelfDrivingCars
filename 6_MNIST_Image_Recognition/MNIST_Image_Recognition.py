#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:26:05 2019

@author: seanxu
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg') # when using 'matplotlib' in python3
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random
np.random.seed(0)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape[0])
assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels."
assert(X_train.shape[1:] == (28,28)), "The dimensions of the images are not 28x28"
assert(X_test.shape[1:] == (28,28)), "The dimensions of the images are not 28x28"
num_of_samples = []
 
cols = 5
num_classes = 10
 
fig, axs = plt.subplots(nrows=num_classes, ncols = cols, figsize=(5, 8))
fig.tight_layout()
for i in range(cols):
    for j in range(num_classes):
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected - 1)), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j))
            num_of_samples.append(len(x_selected))
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
 
y_train = to_categorical(y_train, 10) # create one-hot
y_test = to_categorical(y_test, 10)
 
X_train = X_train/255 # normalization
X_test = X_test/255
 
num_pixels = 784 # do flattening
X_train = X_train.reshape(X_train.shape[0], num_pixels) # (60000,784)
X_test = X_test.reshape(X_test.shape[0], num_pixels) # (10000,784)

def create_model(): # build a regular deep neural network
    model = Sequential() 
    model.add(Dense(10, input_dim=num_pixels, activation='relu')) # input=784 pixels, 10 nodes
    model.add(Dense(30, activation='relu')) #b 30 nodes
    model.add(Dense(10, activation='relu'))
    model.add(Dense(num_classes, activation='softmax')) # output layer, "softmax" converts scores to probabilities
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy']) # configure toe model
    return model
 
 
model = create_model()
print(model.summary())

# train the model
history = model.fit(X_train, y_train, validation_split=0.1, epochs = 10, batch_size = 200, verbose = 1, shuffle = 1)

# plot the training process
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.title('Loss')
plt.xlabel('epoch')
 
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.title('Accuracy')
plt.xlabel('epoch')
 
# evalutaion using test data 
score = model.evaluate(X_test, y_test, verbose=0) # "verbose=0" not display the progress bar
print(type(score)) #"<class 'list'>"
print('Test score:', score[0])
print('Test accuracy:', score[1])
 
import requests
from PIL import Image
 
url = 'https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png'
response = requests.get(url, stream=True)
img = Image.open(response.raw)
plt.imshow(img, cmap=plt.get_cmap('gray')) # (850, 850, 4)
 
import cv2

img = np.asarray(img)
img = cv2.resize(img, (28, 28))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.bitwise_not(img) # convert 255 with 0
plt.imshow(img, cmap=plt.get_cmap('gray'))
 
img = img/255
img = img.reshape(1, 784)
 
prediction = model.predict_classes(img)
print("predicted digit:", str(prediction))