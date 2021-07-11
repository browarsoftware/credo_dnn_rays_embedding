"""
Author: Tomasz Hachaj, 2021
Department of Signal Processing and Pattern Recognition
Institute of Computer Science in Pedagogical University of Krakow, Poland
https://sppr.up.krakow.pl/hachaj/
Data source:
https://credo.nkg-mn.com/hits.html
"""

import random
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.callbacks import CSVLogger
from CreateModel import CreateModel_v1, CreateModel_v2, CreateModel_v3, CreateModel_VGG16_pretrianed, CreateModel_VGG16
from utils import create_dir
""""
from keras.preprocessing import image
from keras.models import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras import initializers

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
"""
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
df=pd.read_csv('d:\\dane\\credo\\dane_string.txt')



sample_size = 235
#model_id = 0
#seed_id = 0
seed_val = [
    0, 101,
    542, 1011, 3333, 4321, 6000, 7777, 10111, 15151]

#seed_val = [0, 542, 1011]
#model_val = ["VGG16", "v1", "v2"]
#model_val = ["v1", "v2", "v3", "VGG16", "VGG16_pretrained"]
model_val = ["VGG16_pretrained"]


mm = CreateModel_VGG16_pretrianed()
mm.summary()

path_to_checkpoints = "checkpoints/"
create_dir('results')
create_dir(path_to_checkpoints)
for model_name in model_val:
    path_help = "results/" + model_name
    create_dir(path_help)
    path_help = "checkpoints/" + model_name
    create_dir(path_help)
    for seed_name in seed_val:
        path_help = "results/" + model_name + "/" + str(seed_name)
        create_dir(path_help)
        path_help = "checkpoints/" + model_name + "/" + str(seed_name)
        create_dir(path_help)

#print(train_generator.image_shape)
# checkpoint



#CreateModel_v1, CreateModel_v2, CreateModel_v3, CreateModel_VGG16_pretrianed, CreateModel_VGG16
#"v1", "v2", "v3", "VGG16", "VGG16_pretrained"

epochs = 200
learning_rate_det = 0.1
learning_rate_step = 100

def lr_scheduler(epoch, lr):
    # if epoch == 1:
    #    lr = 0.01
    if epoch % learning_rate_step == 0 and epoch > 0:
        lr = lr * learning_rate_det
    print(lr)
    return lr


model = None
for model_name in model_val:
    for seed_name in seed_val:
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

        my_seed = seed_name
        my_model = model_name

        random.seed(my_seed)
        my_random_sample = random.sample(range(df.shape[0]), sample_size)
        mask = np.ones(df.shape[0], dtype=bool)
        mask[my_random_sample] = False

        mask_not = np.zeros(df.shape[0], dtype=bool)
        mask_not[my_random_sample] = True

        df_train = df[mask]
        df_valid = df[mask_not]

        columns = ["Kropki", "Kreski", "Robaki", "Artefakty"]
        datagen = ImageDataGenerator(rescale=1. / 255.)
        test_datagen = ImageDataGenerator(rescale=1. / 255.)

        if (model_name == "VGG16_pretrained"):
            train_generator = datagen.flow_from_dataframe(
                # dataframe=df[:1800],
                dataframe=df_train,
                directory="d:\\dane\\credo\\png2",
                x_col="id",
                y_col=columns,
                batch_size=32,
                seed=42,
                shuffle=True,
                class_mode="raw",
                target_size=(60, 60),
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)

            test_generator = datagen.flow_from_dataframe(
                # dataframe=df[:1800],
                dataframe=df_train,
                directory="d:\\dane\\credo\\png2",
                x_col="id",
                y_col=columns,
                batch_size=32,
                seed=42,
                shuffle=True,
                class_mode="raw",
                target_size=(60, 60))
        else:
            train_generator = datagen.flow_from_dataframe(
                # dataframe=df[:1800],
                dataframe=df_train,
                directory="d:\\dane\\credo\\png2",
                x_col="id",
                y_col=columns,
                batch_size=32,
                seed=42,
                shuffle=True,
                class_mode="raw",
                color_mode='grayscale',
                target_size=(60, 60),
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)

            test_generator = datagen.flow_from_dataframe(
                # dataframe=df[:1800],
                dataframe=df_train,
                directory="d:\\dane\\credo\\png2",
                x_col="id",
                y_col=columns,
                batch_size=32,
                seed=42,
                shuffle=True,
                class_mode="raw",
                color_mode='grayscale',
                target_size=(60, 60))

        csv_logger = CSVLogger(path_to_checkpoints + "/" + my_model + "/" + str(my_seed) + "/" + my_model + "_" + str(my_seed) + '.log')
        filepath = path_to_checkpoints + '/' + my_model + '/' + str(my_seed) + "/" + my_model + "_" + str(my_seed) + "-{epoch:02d}-{accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max',
                             save_weights_only=True)


        callbacks_list = [checkpoint, LearningRateScheduler(lr_scheduler, verbose=1), csv_logger]
        model.fit(train_generator, validation_data=test_generator, epochs = epochs, callbacks=callbacks_list)

