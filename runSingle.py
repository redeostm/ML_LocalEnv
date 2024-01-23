'''
runSingle.py

Summary:
This script is the main program for the deep learning training (with validation per each epoch too)
by using the 3D V-Net architecture.

Requirements:
- Tensorflow 2.X package with GPU acceleration : for performing deep learning
- H5py package : for managing HDF5 files
- generatorSingle.py at the same directory : for the definition of V-NET architecture

Usage:
python runSingle.py [current directory] [mode (please use 0)] [minimum learning rate] [maximum learning rate]
'''

import os
import tensorflow as tf
from tensorflow import keras,math
from tensorflow.keras import callbacks,optimizers

import numpy as np
import h5py

import sys

# Use generatorSingle.py at the directory passed by the first argument (see Usage)
directory = sys.argv[1]
sys.path.insert(1,directory)
from generatorSingle import Generator


# Define the mode, minimum/maximum learning rates, and the number of samples per each minibatch.
# 1. In this routine, I assume that the dimension of each output file is (number of quantities, Ndim, Ndim, Ndim),
#    where quantities could be density, gravitational potential, etc.
#    Then "mode" is index of the quantity that you want to train with this routine, starting from zero.
#    If you only treat a single quantity, then mode = 0.
# 2. I use so-called "triangular cyclic learning rate" to optimize the learning process.
#    See https://pyimagesearch.com/2019/07/29/cyclical-learning-rates-with-keras-and-deep-learning/.
#    You can estimate these values by running findLearningRate.py at the same directory before running this script.
# 3. In usual deep learning, number of samples per minibatch is set to, e.g., 16, 32, 64, or 128.
#    The unusually low value of Nbatch = 6 here is mainly because the deep learning architecture here is very memory-consuming.
#    If the architecture becomes lighter after your modification, you may be able to try higher values.
mode = np.int(sys.argv[2])
min_lr,max_lr = np.float(sys.argv[3]),np.float(sys.argv[4])
Nbatch = 6


# Function for generating minibatches for training & validation samples.
# Inputs :
# - files : a full list of the names of files containing the data
# - permutation : True if you want to randomly mix the data order when creating minibatches, and False if not.
#                 You should set it as True when it is used for the training sample.
# Output : A series of minibatches.
def generator_batches(files,permutation=False):

	Nfile = len(files) # Nfile = number of files
	indxFile = np.arange(Nfile) # indxFile = [0, 1, ..., Nfile-1]
	if permutation == True:
		indxFile = np.random.permutation(indxFile) # Mix the order of indxFile if permutation is True.

	counter = 0 # "counter" will save the index at indxFile to read.
	while True:
		smallIndxFile = indxFile[counter:counter+Nbatch//3] # smallIndxFile : a subset of indxFile containing Nbatch samples. Dividing "3" was introduced because it assumes that each input file contains 3 sets of data --- X/Y/Z rotation. 
		counter += Nbatch//3 # Update "counter" so that it points out the start of the next samples.
		if counter >= Nfile: # If "counter" exceeds "Nfiles" --- that is, if it goes beyond indxFile:
			if permutation == True:
				indxFile = np.random.permutation(indxFile) # If the permuation is True, mix the order of indxFile again!
			counter = 0 # Reset "counter" as zero so that the minibatch will be made from the beginning of the list again.

		# Define the shape of input and output arrays in the minibatch.
		# Here, the format is (number of samples, number of quantities, Ndim, Ndim, Ndim).
		# Note that both data are saved with single precision (float, float32) rather than double precision (double, float64) to increase the deep learning performance.
		inputs,outputs = np.zeros((len(smallIndxFile)*3,2,128,128,128),dtype=np.float32),np.zeros((len(smallIndxFile)*3,1,128,128,128),dtype=np.float32)

		# Assign the actual values in the data files to the input and output arrays with iteration.
		for smallIndx in range(len(smallIndxFile)):
			fdata = h5py.File(files[smallIndxFile[smallIndx]],"r") # Open a single data file in HDF5 format (see files[] for its filename).
			tmpInputs,tmpOutputs = fdata['input'][:,:,:,:,:].astype(np.float32),fdata['output'][:,mode,:,:,:].reshape(-1,1,128,128,128).astype(np.float32) # Read input and output data from the file, by assuming that each is saved in "/input/" and "/output/" datasets.
			fdata.close() # Close the HDF5 data file.
			
			indxOrder = np.arange(3) # Order of X/Y/Z rotation : [0, 1, 2]
			if permutation == True:
				indxOrder = np.random.permutation(np.arange(3)) # Mix this too, if permutation is True.
			# Assign the values to inputs[] and outputs[].
			for ssIndx in range(3):
				inputs[ssIndx+3*smallIndx,:,:,:,:] = tmpInputs[indxOrder[ssIndx],:,:,:,:].astype(np.float32)
				outputs[ssIndx+3*smallIndx,:,:,:,:] = tmpOutputs[indxOrder[ssIndx],:,:,:,:].astype(np.float32)

		# Whenever the minibatch is made, put it out!
		yield (inputs,outputs)


Ntrain,Ntest = 10629,1256 # The number of train & validation files. Note that "...test..." means actually validation.
train_files = ["train_set/{}.hdf5".format(inum) for inum in range(Ntrain)] # List of filenames containing training samples.
test_files = ["test_set/{}.hdf5".format(inum) for inum in range(Ntest)] # List of filenames containing validation samples.
gen_train,gen_test = generator_batches(train_files,permutation=True),generator_batches(test_files) # Generators for training & validation samples. Note that "gen_train" uses permutation=True.


# Define "strategy" --- this is necessary especially if you want to use multiple GPUs at once.
strategy = tf.distribute.MirroredStrategy()

# Please define all deep learning-related strategies under "with strategy.scope():"!
with strategy.scope():

	# Define a deep learning model with 3D V-Net architecture --- See generatorSingle.py for details.
	model = Generator()

	# Compile the model with ADAM optimizer and mean-squared-error(MSE) loss function.
	# See https://keras.io/api/optimizers/ for the list of available optimizers.
	# Here, the learning rate at the first epoch is set to the "minimum learning rate" from your input.
	model.compile(optimizer=optimizers.Adam(learning_rate=min_lr),
			loss='mean_squared_error',metrics=['mean_squared_error'])


	# Function for defining triangular cyclical learning rate.
	# This will give a learning rate between minimum and maximum learning rates from your input with 8-epoch period.
	# Input : Epoch number.
	# Output : Learning rate at the given epoch.
	def clr_schedule(epoch):
		num_epoch = 8
		epoch_part = epoch % num_epoch

		mid_pt = int(num_epoch*0.5)
		if epoch_part <= mid_pt:
			return min_lr + epoch_part*(max_lr-min_lr)/mid_pt
		else:
			return max_lr - (epoch_part-mid_pt)*(max_lr-min_lr)/mid_pt


	# Define the learning rate scheduler based on the triangular cyclic learning rate defined just before.
	lr_scheduler = callbacks.LearningRateScheduler(clr_schedule)


	# Define checkpoints --- moments to save the deep learning models DURING the training.
	checkpointMinValLoss = callbacks.ModelCheckpoint("minValLoss_mode{}.hdf5".format(mode),
			monitor='val_loss',verbose=1,save_best_only=True, mode='auto') # Save the model when the validation loss becomes minimum.
	checkpointMinLoss = callbacks.ModelCheckpoint("minLoss_mode{}.hdf5".format(mode),
			monitor='loss',verbose=1,save_best_only=True, mode='auto') # Save the model when the training loss becomes minimum.
	callbacks_list = [checkpointMinValLoss,checkpointMinLoss,lr_scheduler] 


	# Fit the deep learning model with:
	# - Training dataset will be provided by our custom minibatch generator (gen_train)
	# - Each epoch will cover the entire training dataset (steps_per_epoch = ...)
	# - 200 epochs will be used during the training (epochs = ...)
	# - Print the minimum information per each epoch (verbose = ...)
	# - Validation dataset will be also provided by our custom minibatch generator (validation_data = ...)
	# - Each epoch will also cover the entire validation dataset (validation_steps = ...)
	# - It will save the models during the training when training/validation losses become minimum (callbacks = ...)
	run = model.fit(gen_train,steps_per_epoch=int(Ntest*3/Nbatch),epochs=200,verbose=2,
			validation_data=gen_test,validation_steps=int(Ntest*3/Nbatch),callbacks=callbacks_list)


	# When the training is over, save the model too!
	model.save("final_mode{}.hdf5".format(mode))
