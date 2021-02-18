"""
Code for the model architecture.

Mythcell
"""
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dropout, Dense

def c2_network(ishape,num_cat,compile_model=True):
    model = Sequential(
    [
        keras.Input(shape=ishape),
        Conv2D(32,kernel_size=(7,7),activation='relu'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2,2)),
        Conv2D(64,kernel_size=(5,5),activation='relu'),
        Conv2D(64,kernel_size=(5,5),activation='relu'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2,2)),
        Conv2D(128,kernel_size=(3,3),activation='relu'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2,2)),
        Flatten(),
        Dropout(0.5),
        Dense(256),
        Dense(256),
        Dense(num_cat,activation='softmax')
    ])
    if compile_model:
        # default tuned learning rate is 2e-4
        model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(2e-4),metrics=["accuracy"])
    return model