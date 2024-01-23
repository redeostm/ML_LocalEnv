'''
findLearningRate.py

Summary:
This script helps the user finding out a proper range of learning rate by using the 3D V-Net architecture.
This is mostly a simplified version of runSingle.py, except last a few lines.

Requirements:
- Tensorflow 2.X package with GPU acceleration : for performing deep learning
- H5py package : for managing HDF5 files
- generatorSingle.py at the same directory : for the definition of V-NET architecture

Usage:
python runSingle.py [current directory] [mode (please use 0)]
'''

#!/home/swhong/anaconda3/envs/tf2/bin/python

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks

import numpy as np
import h5py

import sys

directory = sys.argv[1]
sys.path.insert(1,directory)
from generatorSingle import Generator

mode = np.int(sys.argv[2])
Nbatch = 6

# Data preparation
def generator_batches(files,permutation=False):
	Nfile = len(files)
	indxFile = np.arange(Nfile)
	if permutation == True:
		indxFile = np.random.permutation(indxFile)

	counter = 0
	while True:
		smallIndxFile = indxFile[counter:counter+Nbatch//3]
		counter += Nbatch//3
		if counter >= Nfile:
			if permutation == True:
				indxFile = np.random.permutation(indxFile)
			counter = 0

		inputs,outputs = np.zeros((len(smallIndxFile)*3,2,128,128,128),dtype=np.float32),np.zeros((len(smallIndxFile)*3,1,128,128,128),dtype=np.float32)

		for smallIndx in range(len(smallIndxFile)):
			fdata = h5py.File(files[smallIndxFile[smallIndx]],"r")
			tmpInputs,tmpOutputs = fdata['input'][:,:,:,:,:].astype(np.float32),fdata['output'][:,mode,:,:,:].reshape(-1,1,128,128,128).astype(np.float32)
			fdata.close()
			
			indxOrder = np.random.permutation(np.arange(3))
			for ssIndx in range(3):
				inputs[ssIndx+3*smallIndx,:,:,:,:] = tmpInputs[indxOrder[ssIndx],:,:,:,:].astype(np.float32)
				outputs[ssIndx+3*smallIndx,:,:,:,:] = tmpOutputs[indxOrder[ssIndx],:,:,:,:].astype(np.float32)

		yield (inputs,outputs)


Ntrain,Ntest = 10629,1256
train_files = ["train_set/{}.hdf5".format(inum) for inum in range(Ntrain)]
test_files = ["test_set/{}.hdf5".format(inum) for inum in range(Ntest)]
gen_train,gen_test = generator_batches(train_files,permutation=True),generator_batches(test_files)


# Set model, loss function, and optimizer
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
	model = Generator()

	# The main difference between this code and runSingle.py starts here.
	# Here, each epoch contains only one minibatch. 
	# After each epoch, the learning rate is changed from 10^(-9) to 10^1 with 10^(0.1) logarithmic interval.
	# Then the train/validation losses will be updated for each epoch.
	# See https://pyimagesearch.com/2019/08/05/keras-learning-rate-finder/ to understand how to estimate a proper range of learning rate with this.
	for nIter,learningRate in enumerate(np.logspace(-9,1,91)):
		model.compile(optimizer=keras.optimizers.Adam(learning_rate=learningRate),loss='mean_squared_error',metrics=['mean_squared_error'])

		print("== learning rate = {:.5G} ==".format(learningRate))
		run = model.fit(gen_train,steps_per_epoch=int(6*3/Nbatch),epochs=1,verbose=2)
