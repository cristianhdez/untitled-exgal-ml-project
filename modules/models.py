from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input

from keras.layers import LeakyReLU


from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

def create_mlp(dim):
    # define our MLP network
    model = Sequential()
    model.add(Dense(32, input_dim=dim, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation = "linear"))
    # return our model
    return model

def create_mlp2(dim):
    # define our MLP network
    model = Sequential()
    
    model.add(Dense(1024, input_dim=dim, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    # return our model
    return model

def create_cnn(dim):
    # define our MLP network
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(dim,1)))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])