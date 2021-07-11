# credo_dnn_rays_embedding
This is a source code for the paper "Deep neural network architecture for low-dimensional embedding and classification cosmic ray images obtained from CMOS cameras" 
by Tomasz Hachaj, Marcin Piekarczyk and Łukasz Bibrzycki. Source codes written by Tomasz Hachaj.

Requires: Python 3.6, Keras 2.4, Tensorflow 2.3 (GPU), JAVA for classification.

Train_model.py - main file for training DNN.

Generate_results.py - applies DNN to make embedding.

MakeClassification.java - uses files from Generate_results.py to make classification.

In order to run project download following files and change paths in source codes.

Link to CREDO dataset [LINK](https://user.credo.science/user-interface/download/images/)

Link to training data set (images) [LINK](https://www.dropbox.com/s/5ye4zddp0r8gqgz/CREDOTrainingData.zip?dl=0)

Link to training dataset (labels) [LINK](https://www.dropbox.com/s/ozsp5r2hnb8rvfl/dane_string.txt?dl=0)

Link to Results [LINK](https://www.dropbox.com/s/lwv2jmk9ktfw9u1/all.zip?dl=0)

Link to trained weights [LINK](https://www.dropbox.com/sh/tc5rlcrjbrcbrmh/AABKBaRJyR0pilCNfYMbe-INa?dl=0)

