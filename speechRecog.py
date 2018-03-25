# Test Speech signal processing with 
# Deep Neural Network->Long-Short Term Memorty (Tensorflow)
# Mr.Suradej Duangpummet
# Last Update : March 21,2018

from __future__ import division, print_function, absolute_import
import os
import tflearn
import tensorflow as tf
import speech_data_mod
import time
from datetime import datetime

#---- Hyperparameters
learning_rate = 0.0001
training_iters = 5000#3000000  # steps
batch_size = 64
width = 20  # MFCC features : 
height = 80  # (max) length of utterance
classes = 10  # digits

#Split Data for train and test
batch = word_batch = speech_data_mod.mfcc_batch_generator(batch_size)
X, Y = next(batch)
trainX, trainY = X, Y
testX, testY = X, Y 

# Network building
net = tflearn.input_data([None, width, height])
net = tflearn.lstm(net, 128, dropout=0.5)	#net = tflearn.lstm(net, 128*4, dropout=0.5)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

## add this "fix" for tensorflow version errors
col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for x in col:
	tf.add_to_collection(tf.GraphKeys.VARIABLES, x )

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)

for step in range(0, training_iters):
	print("fitting step:",step)
	trainX, trainY = next(batch)
	testX, testY = next(batch)  # todo: proper ;)	
	model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True, batch_size=batch_size)	


model.save("myASR-model_2") #save model

