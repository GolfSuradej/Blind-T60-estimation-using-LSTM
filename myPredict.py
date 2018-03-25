import os
import re
import sys
import wave

import numpy
import numpy as np
import skimage.io  # scikit-image
import librosa

import tflearn
import speech_data_mod

batch_features = []

#Hyperparameters
learning_rate = 0.0001
training_iters = 50#3000000  # steps
batch_size = 64
width = 20  # MFCC features : Mel-Frequency Cepstral Coefficients
height = 80  # (max) length of utterance
classes = 10  # digits


test_file = "0_0Test_Junior100.wav"
#test=speech_data_mod.load_wav_file(speech_data_mod.path + test_file)

wave, sr = librosa.load(speech_data_mod.path + test_file, mono=True)

mfcc = librosa.feature.mfcc(wave, sr)

mfcc=np.pad(mfcc,((0,0),(0,80-len(mfcc[0]))), mode='constant', constant_values=0)

batch_features.append(np.array(mfcc))

#print("\n MFCC shape:",np.array(mfcc).shape)
#print("\n batch feature",np.array(batch_features).shape)


net = tflearn.input_data([None, width, height])
net = tflearn.lstm(net, 128, dropout=0.5)	#net = tflearn.lstm(net, 128*4, dropout=0.5)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_verbose=0)
# Load a model
model.load("myASR-model_1")

result=model.predict(batch_features)

result=numpy.argmax(result)
print("predicted digit result = %d ",result)
