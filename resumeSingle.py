import os
import tensorflow as tf
from tensorflow import keras,math
from tensorflow.keras import callbacks,optimizers

import numpy as np
import h5py

import sys

directory = sys.argv[1]
sys.path.insert(1,directory)
from generatorSingle import Generator

mode = np.int(sys.argv[2])
min_lr,max_lr = np.float(sys.argv[3]),np.float(sys.argv[4])
startFile = sys.argv[5]
startEpoch,endEpoch = np.int(sys.argv[6]),np.int(sys.argv[7])
Nbatch = 8

# Data preparation
def generator_batches(files,permutation=False):
	Nfile = len(files)
	indxFile = np.arange(Nfile)
	if permutation == True:
		indxFile = np.random.permutation(indxFile)

	counter = 0
	while True:
		smallIndxFile = indxFile[counter:counter+Nbatch]
		counter += Nbatch
		if counter >= Nfile:
			if permutation == True:
				indxFile = np.random.permutation(indxFile)
			counter = 0

		inputs,outputs = np.zeros((len(smallIndxFile),2,128,128,128),dtype=np.float32),np.zeros((len(smallIndxFile),1,128,128,128),dtype=np.float32)

		for smallIndx in range(len(smallIndxFile)):
			fdata = h5py.File(files[smallIndxFile[smallIndx]],"r")
			inputs[smallIndx,:,:,:,:] = fdata['input'][0,:,:,:,:].astype(np.float32)
			outputs[smallIndx,:,:,:,:] = fdata['output'][0,mode,:,:,:].reshape(1,1,128,128,128).astype(np.float32)
			fdata.close()

		yield (inputs,outputs)


Ntrain,Ntest = 10629,1256
train_files = ["train_set/{}.hdf5".format(inum) for inum in range(Ntrain)]
test_files = ["test_set/{}.hdf5".format(inum) for inum in range(Ntest)]
gen_train,gen_test = generator_batches(train_files,permutation=True),generator_batches(test_files)


# Set model, loss function, and optimizer
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
	model = Generator()
	model.load_weights(startFile)

	model.compile(optimizer=optimizers.Adam(learning_rate=min_lr),
			loss='mean_squared_error',metrics=['mean_squared_error'])


	# Define triangular cyclical learning rate
	def clr_schedule(epoch):
		num_epoch = 8
		epoch_part = epoch % num_epoch

		mid_pt = int(num_epoch*0.5)
		if epoch_part <= mid_pt:
			return min_lr + epoch_part*(max_lr-min_lr)/mid_pt
		else:
			return max_lr - (epoch_part-mid_pt)*(max_lr-min_lr)/mid_pt

	lr_scheduler = callbacks.LearningRateScheduler(clr_schedule)

	# Define checkpoint
	checkpointMinValLoss = callbacks.ModelCheckpoint("minValLoss_mode{}.hdf5".format(mode),
			monitor='val_loss',verbose=1,save_best_only=True, mode='auto')
	checkpointMinLoss = callbacks.ModelCheckpoint("minLoss_mode{}.hdf5".format(mode),
			monitor='loss',verbose=1,save_best_only=True, mode='auto')
	callbacks_list = [checkpointMinValLoss,checkpointMinLoss,lr_scheduler]


	# Run it!
	run = model.fit(gen_train,steps_per_epoch=int(Ntest/Nbatch),epochs=endEpoch-startEpoch,verbose=2,
			validation_data=gen_test,validation_steps=int(Ntest/Nbatch),callbacks=callbacks_list)
	model.save("final_mode{}.hdf5".format(mode))
