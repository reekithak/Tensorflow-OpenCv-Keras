import os 
import pandas as pd
import numpy as np
import keras 
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense , LeakyReLU , ELU , Dropout ,Flatten , BatchNormalization , Activation
from keras.activations import relu , sigmoidfrom keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
def hyper_tune_model(layers,activation):
    model = Sequential()
    for i , nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes,input_dim = X_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
    model.add(Dense(units=1,kernel_initializer = 'glorot_uniform',activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model
model = KerasClassifier(build_fn = hyper_tune_model,verbose=1)
layers = [(20),(40,20)]
activation = ['sigmoid','relu']
parameter_grid = dict(layers=layers,activation=activation,batch_size=(128,256),epochs=(10,30))
grid = GridSearchCV(estimator = model , param_grid=parameter_grid,cv=5)
grid_result = grid.fit(X_train,y_train)