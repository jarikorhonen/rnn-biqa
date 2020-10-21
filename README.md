# RNN-based blind image quality model (RNN-BIQA)

RNN-BIQA model divides input image in patches and then uses a CNN model to extract features from each patch to form a sequence of feature vectors. Then, RNN-based model is employed to predict subjective quality score (MOS) from the feature vector sequence. 

Training the model includes two phases. First, the CNN model for feature extraction needs to be trained on image patches from the training set. We provide a pre-trained CNN feature extractor [here](https://mega.nz/file/jYZhxIqQ#lddqs1ne_d7ILfbMP7CNzkNHdJ3q6sHjgL0twH3GvXo). For advice how to train CNN feature extractor, you can refer to the GitHub page of CNN-TLVQM video quality model. Note that the CNN models used in CNN-TLVQM and RNN-BIQA are similar, but not identical.

For the second phase, training and testing the RNN-based quality model, you can refer to Matlab code example in *KoNIQ_Example.mat*. To use this example, you need to have KoNIQ-10k image quality downloaded, available in http://database.mmsp-kn.de/koniq-10k-database.html.

The code has been tested using Matlab version R2020a. Note that GRU layers are not supported in earlier versions of Matlab.
