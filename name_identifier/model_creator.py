import os
import sys
import numpy as np
import pandas as pd
import random
import string
import pickle
import bz2

import keras
from keras import layers
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

class model_creator_class():
    def __init__(self, epoch, batch_size, val_split, acc_metric,lr):
        work_path = "D:/Chrome_downloads/name_identifier/"
        self.model_path = work_path + "models/"
        self.data_path = work_path + "data/"
        self.epoch = epoch
        self.batch_size = batch_size
        self.val_split = val_split
        self.accuracy_metric = acc_metric
        self.lr = lr
        
    def data_loader(self, data):
        _read =  bz2.BZ2File(self.data_path + data,"rb")
        loaded_var_name=pickle.load(_read)
        _read.close()
        
        return loaded_var_name
    
    def model(self,datax,datay,slide_kernel):
        model = Sequential()
        model.add(LSTM(256, input_shape=(datax.shape[1], datax.shape[2]),return_sequences=True))
#         model.add(Dropout(0.2))
        model.add(LSTM(256))
        model.add(Dense(datay.shape[1], activation='softmax'))
        optimizer = keras.optimizers.Adam(lr=self.lr, decay=0.000001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=[self.accuracy_metric])
        
        filepath=self.model_path + "weights_improvement_" + str(slide_kernel) + "_{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        model.fit(datax, datay, epochs=self.epoch, batch_size=self.batch_size, validation_split = self.val_split, shuffle = True, callbacks=callbacks_list)
        
        return None