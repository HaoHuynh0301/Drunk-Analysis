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


#dimensions of our images
base_path = '/content/drive/My Drive/Colab Notebooks/Drunk_Detection/models'
dataset_path='/content/drive/My Drive/Colab Notebooks/Drunk_Detection/dataset_csv/training_dataset_sorted.csv'
img_width, img_height = 48, 48
patience = 50
EPOCHS=50
BATCH_SIZE=32
input_shape=(48,48,1)

#loading dataset
faces, emotions=load_data(dataset_path)
faces=preprocess_input(faces)
num_samples, num_classes = emotions.shape
xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)
# x_data, y_data = read_data("/content/drive/My Drive/Colab Notebooks/Drunk_Detection/dataset_csv/drunk.csv", "/content/drive/My Drive/Colab Notebooks/Drunk_Detection/dataset_csv/normal.csv")
# x_data[x_data == True] = 1
# x_data[x_data == False] = 0
# from sklearn.utils import shuffle
# (x_data, y_data) = shuffle(x_data, y_data)
# xtrain, xtest, ytrain, ytest = train_test_split(x_data, y_data)

# data generator
data_generator = ImageDataGenerator(
                            rescale=1./255,
                            shear_range=0.1,
                            zoom_range=0.1,
                            horizontal_flip=True)

validation_data_generator = ImageDataGenerator(rescale=1./255)

training_generator = data_generator.flow(
                            xtrain, ytrain,
                            batch_size=BATCH_SIZE)

validation_generator=validation_data_generator.flow(
                            xtest, ytest,
                            batch_size=BATCH_SIZE
)


print(xtrain.shape)


#build CNN
input_shape=(48, 48, 1)
model=keras.Sequential()

model.add(Conv2D(8, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))

model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())

model.add(Dense(64))
model.add(Dropout(0.25))

model.add(Dense(2))
model.add(Activation('softmax'))

# model.compile(loss='binary_crossentropy', # or categorical_crossentropy
#               optimizer='rmsprop',
#               metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit_generator(
                  training_generator,
                  steps_per_epoch=len(xtrain) / BATCH_SIZE,
                  validation_data=validation_generator,
                  validation_steps=len(xtest) / BATCH_SIZE,
                  epochs=EPOCHS,verbose=1)

# train_history = model.fit(xtrain, ytrain, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = 0.2)

model.save('model_drunk.h5')
