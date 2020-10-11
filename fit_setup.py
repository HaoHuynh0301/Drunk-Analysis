import keras
import pandas as pd
import numpy as np
import matplotlib as mpl
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from load_datacsv import load_data, preprocess_input
from sklearn.model_selection import train_test_split


# dimensions of our images.
img_width, img_height = 48, 48
input_shape=(48,48,1)

#loading dataset
dataset_path='/Users/macos/Downloads/training_dataset.csv'
faces, emotions=load_data(dataset_path)
faces=preprocess_input(faces)
num_samples, num_classes = emotions.shape
xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions)

# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)


#build CNN
model=keras.Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', # or categorical_crossentropy
              optimizer='rmsprop',# or adagrad
              metrics=['accuracy'])

model.fit_generator(data_generator.flow(xtrain, ytrain,
                                            32),
                        steps_per_epoch=len(xtrain),
                        epochs=50, verbose=1,
                        validation_data=(xtest,ytest))

model.save('model_drunk.h5')