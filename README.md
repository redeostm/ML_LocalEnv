# ML_LocalEnv
This is a codelet used for reconstructing the dark matter density of the local universe from galaxy position and radial peculiar velocity via 3D V-Net deep learning.

See Hong, S.E., Jeong, D., Hwang, H.S. & Kim, J., *Revealing the Local Cosmic Web from Galaxies by Deep Learning*, ApJ 913, 76 (2021) [https://ui.adsabs.harvard.edu/abs/2021ApJ...913...76H/abstract]

## Structure
- generatorSingle.py : Defines the 3D V-Net deep learning architecture
- findLearningRate.py : Helps the users find out a proper range of learning rates. See [https://pyimagesearch.com/2019/08/05/keras-learning-rate-finder/]
- runSingle.py : Main code for training
- resumeSingle.py : Same as runSingle.py, but it uses resuming the training from a saved hyperparameter file 
