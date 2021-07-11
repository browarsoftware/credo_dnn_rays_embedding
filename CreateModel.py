"""
Author: Tomasz Hachaj, 2021
Department of Signal Processing and Pattern Recognition
Institute of Computer Science in Pedagogical University of Krakow, Poland
https://sppr.up.krakow.pl/hachaj/
Data source:
https://credo.nkg-mn.com/hits.html
"""

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

def CreateModel_VGG16_pretrianed():
    hidden_layer_neurons_count = 128
    output_layer_neurons_count = 4

    model = VGG16(weights='imagenet', include_top=False, input_shape=(60, 60, 3))

    for l in model.layers:
        l.trainable = False

    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(hidden_layer_neurons_count, activation='relu')(flat1)
    output = Dense(output_layer_neurons_count, activation='softmax')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
    """
    x = Flatten()(model.get_layer("block5_pool").output)
    x = Dense(hidden_layer_neurons_count, activation="relu")(x)
    x = Dense(output_layer_neurons_count, activation='sigmoid')(x)

    model = Model(inputs=model.inputs,
                   outputs=x)
    """
    #model2.
    """
    model = Sequential().add(model)

    model.add(Flatten())
    model.add(Dense(hidden_layer_neurons_count, activation="relu"))
    model.add(Dense(output_layer_neurons_count, activation='sigmoid'))
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
    """
    return model

def CreateModel_VGG16():
    hidden_layer_neurons_count = 128
    output_layer_neurons_count = 4
    model = Sequential()
    model.add(Conv2D(input_shape=(60,60,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(hidden_layer_neurons_count, activation="relu"))
    model.add(Dense(output_layer_neurons_count, activation='sigmoid'))
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def CreateModel_v1():
    model = Sequential()
    model.add(Conv2D(input_shape=(60, 60, 1), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=8, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(8, 8), strides=(8, 8)))

    hidden_layer_neurons_count = 16
    output_layer_neurons_count = 4

    model.add(Flatten())
    model.add(Dense(hidden_layer_neurons_count, activation="relu"))
    model.add(Dense(output_layer_neurons_count, activation='sigmoid'))#linear
    #model.add(Dense(output_layer_neurons_count, activation='linear'))  # linear
    #model.add(Dense(output_layer_neurons_count, activation='relu'))  # linear
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def CreateModel_v2():
    model = Sequential()
    model.add(Conv2D(input_shape=(60, 60, 1), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(8, 8), strides=(8, 8)))

    hidden_layer_neurons_count = 16
    output_layer_neurons_count = 4

    model.add(Flatten())
    model.add(Dense(hidden_layer_neurons_count, activation="relu"))
    model.add(Dense(output_layer_neurons_count, activation='sigmoid'))
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def CreateModel_v3():
    model = Sequential()
    model = Sequential()
    model.add(Conv2D(input_shape=(60,60,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(4, 4), strides=(4, 4)))

    hidden_layer_neurons_count = 16
    output_layer_neurons_count = 4

    model.add(Flatten())
    model.add(Dense(hidden_layer_neurons_count, activation="relu"))
    model.add(Dense(output_layer_neurons_count, activation='sigmoid'))
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
    return model