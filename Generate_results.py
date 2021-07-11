"""
Author: Tomasz Hachaj, 2021
Department of Signal Processing and Pattern Recognition
Institute of Computer Science in Pedagogical University of Krakow, Poland
https://sppr.up.krakow.pl/hachaj/
Data source:
https://credo.nkg-mn.com/hits.html
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
# %matplotlib notebook
from keras.preprocessing import image
import keras.backend as K
from CreateModel import CreateModel_v1, CreateModel_v2, CreateModel_v3, CreateModel_VGG16_pretrianed, CreateModel_VGG16
import random

sample_size = 235

#model_val = ["v1", "v2", "v3", "VGG16_pretrained"]
model_val = ["VGG16_pretrained"]
seed_val = [0, 101, 542, 1011, 3333, 4321, 6000, 7777, 10111, 15151]
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.models import Model
import tensorflow as tf

import glob
import os

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

############################################
data = load_breast_cancer()
df=pd.read_csv('d:\\dane\\credo\\dane_string.txt')

path_to_checkpoints = "checkpoints/"

for model_name in model_val:
    for seed_name in seed_val:

        my_model = model_name
        my_seed = seed_name
        print(str(model_name) + "," + str(my_seed))
        #path_help = "results/" + model_name + "/" + str(seed_name)
        my_path = path_to_checkpoints + '/' + my_model + '/' + str(my_seed) + "/"

        list_of_files = glob.glob(my_path + "/" + '*.hdf5')  # * means all if need specific format then *.hdf5
        latest_file = max(list_of_files, key=os.path.getctime)
        import numpy as np
        from numpy import genfromtxt

        if (model_name == "v1"):
            model = CreateModel_v1()
        if (model_name == "v2"):
            model = CreateModel_v2()
        if (model_name == "v3"):
            model = CreateModel_v3()
        if (model_name == "VGG16"):
            model = CreateModel_VGG16()
        if (model_name == "VGG16_pretrained"):
            model = CreateModel_VGG16_pretrianed()
        # flatten
        model.load_weights(latest_file)



        random.seed(my_seed)
        my_random_sample = random.sample(range(df.shape[0]), sample_size)
        mask = np.ones(df.shape[0], dtype=bool)
        mask[my_random_sample] = False

        mask_not = np.zeros(df.shape[0], dtype=bool)
        mask_not[my_random_sample] = True

        df_train = df[mask]
        df_valid = df[mask_not]

        for a in range(len(my_random_sample)):
            id_help = my_random_sample[a]
            file_object = open( "results/" + model_name + '/res'+ "_" + my_model + "_" + str(my_seed) + '.txt', 'a')
            #d:\\dane\\credo\\png2
            img_path = 'd:\\dane\\credo\\png2\\' + df.id[id_help]
            #print(img_path)

            if (model_name == "VGG16_pretrained"):
                img = image.load_img(img_path,
                                     target_size=(60, 60))
            else:
                img = image.load_img(img_path,
                                     color_mode="grayscale",
                                     target_size=(60, 60))
            x = image.img_to_array(img)
            x = x * 1. / 255.
            x = np.expand_dims(x, axis=0)
            # x = preprocess_input(x)

            # my_x = my_x.reshape(1, my_x.shape[0])
            #features = model_flatten(x)[0]
            features = model(x)[0]
            features = K.eval(features)
            #print(features)
            str_help = str(df.id[id_help]) + "," + str(df.Kropki[id_help]) + "," + str(df.Kreski[id_help]) + "," + str(df.Robaki[id_help]) + "," + str(df.Artefakty[id_help])
            for b in range(len(features)):
                str_help = str_help + "," + str(features[b])
            #print(str_help)
            file_object.write(str_help)
            file_object.write('\n')
            file_object.close()
