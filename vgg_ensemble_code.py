#Importing modules.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input

#No of classes present.
print("\nNo of classes: ", len(os.listdir("/home/cse9040/Ishan/Dataset1/Data1/train")))

#Image Generators definition.
train_datagen = ImageDataGenerator(rotation_range = 360, width_shift_range = 0.3, height_shift_range = 0.3, zoom_range = 0.5, shear_range = 0.3, horizontal_flip = True, vertical_flip = True, preprocessing_function = preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

#Loading Dataset.
train = train_datagen.flow_from_directory(directory = "/home/cse9040/Ishan/Dataset1/Data1/train", target_size = (256, 256), batch_size = 40)
val = val_datagen.flow_from_directory(directory = "/home/cse9040/Ishan/Dataset1/Data1/valid", target_size = (256, 256), batch_size = 40)

#Printing image shape.
t_img, label = train.next()
print("\nImage shape: ", t_img.shape)

#New Ensemble Model.
num_models = 3

ensemble_models = []

for i in range(num_models):
    base_model = VGG19(input_shape = (256, 256, 3), include_top = False)
    
    for layer in base_model.layers:
        layer.tranable = False

    X = Flatten()(base_model.output)
    X = Dense(units = 38, activation = 'softmax')(X)

    model = Model(base_model.input, X)

    model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ['accuracy'])

    ensemble_models.append(model)

cnt = 1
for model in ensemble_models:
    es = EarlyStopping(monitor = 'val_accuracy', min_delta = 0.01, patience = 3, verbose = 1)

    filepath = "/home/cse9040/Models/" + "best_model" + str(cnt) + ".h5"
    cnt += 1

    mc = ModelCheckpoint(filepath = filepath, monitor = 'val_accuracy', min_delta = 0.01, patience = 3, verbose = 1, save_best_only = True)

    cb = [es, mc]

    model.fit_generator(train, steps_per_epoch = 30, epochs = 50, verbose = 1, callbacks = cb,  validation_data = val, validation_steps = 30)

