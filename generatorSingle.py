'''
generatorSingle.py

Summary:
This script defines the 3D V-Net deep learning architecture.

Requirements:
- Tensorflow 2.X package with GPU acceleration : for performing deep learning

Usage:
findLearingRate.py, runSingle.py, and resumeSingle.py import Generator() in this script.
'''

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# A class defining the 3D V-Net deep learning architecture.
def Generator():

	# Alias of the generalized layers at the encoding phase.
	# Inputs :
	# - inputLayer : Input of the entire layer.
	# - Nchannel : Number of channels one wants to have at the end of this layer.
	# - batchNorm : True if one wants to perform batch normalization to its input.
	# Output : Output of the entire layer.
	def encodeLayer(inputLayer,Nchannel,batchNorm=True):

		# Step 1. "inputs" = input of the entire layer.
		inputs = layers.BatchNormalization()(inputLayer) # Perform batch normalization if batchNorm is True.
		if batchNorm == False:
			inputs = inputLayer

		# Step 2. Apply 2-pixel reflection paddings on both side, in all three dimensions.
		pad = layers.Lambda(lambda x: tf.pad(x,[[0,0],[0,0],[2,2],[2,2],[2,2]],'REFLECT'))(inputs)

		# Step 3. Apply 3-dimensional convolution with 5x5x5 kernel, 2-pixel strides.
		#         Then the size of spatial dimension will be half.
		#         After the convolution, apply the rectified linear unit (ReLU) activation function to add nonlinearity.
		conv = layers.Conv3D(Nchannel,5,strides=2,padding='valid',activation='relu',data_format="channels_first")(pad)

		return conv


	# Alias of the generalized layers at the decoding phase.
	# Inputs :
	# - inputLayer : Input of the entire layer.
	# - concatLayer : An output from the encoding phase with the same size of spatial dimension with the output of this layer.
	#                 It will be used for concatenation process to minimize the loss of small-scale information.
	# - Nchannel : Number of channels one wants to have at the end of this layer.
	# - activation : Name of the activation function.
	# Output : Output of the entire layer.
	def decodeLayer(inputLayer,concatLayer,Nchannel,activation='relu'):

		# Step 1. Upsampling the input so that the size of spatial dimension becomes double.
		up = layers.UpSampling3D(data_format="channels_first")(inputLayer)

		# Step 2. Concatenate with an output from the encoding phase, which minimizes the loss of small-scale information.
		concat = layers.Concatenate(axis=1)([up,concatLayer])

		# Step 3. Perform batch normalization.
		batch = layers.BatchNormalization()(concat)

		# Step 4. Apply 1-pixel reflection paddings on both side, in all three dimensions.
		pad = layers.Lambda(lambda x: tf.pad(x,[[0,0],[0,0],[1,1],[1,1],[1,1]],'REFLECT'))(batch)

		# Step 5. Apply 3-dimensional convolution with 3x3x3 kernel, 1-pixel stride.
		#         Size of sptial dimension does not change during the covolution --- which is double of the input.
		#         After the convolution, apply the given activation function.
		conv = layers.Conv3D(Nchannel,3,padding='valid',activation=activation,data_format="channels_first")(pad)

		return conv



	# The actual definition of 3D V-Net deep learning architecture starts here!

	# Step 1. Encoding phase
	inputs = keras.Input(shape=(2,128,128,128)) # Input data with two quantities (galaxy number/radial peculiar velocity) and 128^3 grids.
  
	conv64 = encodeLayer(inputs,128,batchNorm=False) # (128,64,...)
	conv32 = encodeLayer(conv64,256) # (256,32,...)
	conv16 = encodeLayer(conv32,512) # (512,16,...)
	conv08 = encodeLayer(conv16,1024) # (1024,8,...)
	conv04 = encodeLayer(conv08,2048) # (2048,4,...)


	# Step 2. Decoding phase
	decv08 = decodeLayer(conv04,conv08,1024) # (1024,8,...)
	decv16 = decodeLayer(decv08,conv16,512) # (512,16,...)
	decv32 = decodeLayer(decv16,conv32,256) # (256,32,...)
	decv64 = decodeLayer(decv32,conv64,128) # (128,64,...)

	outputs = decodeLayer(decv64,inputs,1,activation='tanh') # (1,128,...) : Output data with one quantity and 128^3 grids. Note that the activation function is tanh rather than ReLU so that the values are between -1 and +1.


	# Returns the architecture that uses inputs[] and outputs[] as deep learning input and output.
	return keras.Model(inputs=inputs,outputs=outputs)
