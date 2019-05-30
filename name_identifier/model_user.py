import os
import sys
import numpy as np
import pandas as pd
import keras
import random
import string
import pickle
import bz2
from keras.models import load_model

class model_user_class():
    def __init__(self,model_name,slide_kernel):
        work_path = "D:/Chrome_downloads/name_identifier/"
        self.model_path = work_path + "models/"
        self.data_path = work_path + "data/"
        self.model_name = model_name
        self.slide_kernel = slide_kernel
        self.model = load_model(self.model_path+self.model_name)
        
        # max_len (as got from names array while data creation)
        
        self.max_name_len_ext = 68
        
    def index_2_char(self, index):
        if index==26:
            char = " "
        else:
            char = chr(index + 97)
        return char
    
    def create_names_matrix(self,x):
        names_matrix = np.zeros((self.max_name_len_ext,27))
        for i in range(len(x)):
            try:
                names_matrix[i][string.ascii_lowercase.index(x[i])]=1
            except:
                names_matrix[i][-1]=1
        return names_matrix
    
    def get_name_data(self,x):
        self.data_dict = {}
        self.name_matrix = self.create_names_matrix(x)
        datax = []
        datay = []
        for i in range(0, self.max_name_len_ext - self.slide_kernel, 1):
            seq_in = self.name_matrix[i:i + self.slide_kernel]
            if np.count_nonzero(seq_in)!=0:
                seq_out = self.name_matrix[i + self.slide_kernel]
                datax.append(seq_in)
                datay.append(seq_out)
            else:
                pass
        self.data_dict["data_x"] = np.array(datax)
        self.data_dict["data_y"] = np.array(datay)
        
        return self.data_dict
    
    def get_prediction(self,z):
        xdata = self.get_name_data(z)["data_x"]
        prediction = self.model.predict(xdata,verbose=0)
        char_ = self.index_2_char(np.argmax(prediction[0]))
        return char_
    
    def char2index(self, z):
        try:
            ind = string.ascii_lowercase.index(z)
        except:
            ind = 26
        return ind
    
    def data_loader(self, data):
        _read =  bz2.BZ2File(self.data_path + data,"rb")
        loaded_var_name=pickle.load(_read)
        _read.close()
        
        return loaded_var_name
            
    def get_name_pred_prob(self,z):
        name_len = len(z)
        j=0
        self.name_pred_prob = 1
        while j < name_len - self.slide_kernel:
            name_init = z[j:j+self.slide_kernel]
            pred_init = z[j+self.slide_kernel]
#             print(name_init, pred_init)
            pred_prob = self.model.predict(self.get_name_data(name_init)["data_x"],verbose=0)[0][self.char2index(pred_init)]
            try:
                self.name_pred_prob = self.name_pred_prob*pred_prob
            except:
                pass
#             print(pred_prob)
            j+=1
        return self.name_pred_prob
    
    def get_name_pred_info_gain_prob(self,z):
        name_len = len(z)
        j=0
        inf_gain_tot = 0
        self.inf_gain_prob = 1
        while j < name_len - self.slide_kernel:
            name_init = z[j:j+self.slide_kernel]
            pred_init = z[j+self.slide_kernel]
#             print(name_init, pred_init)
            pred_prob = self.model.predict(self.get_name_data(name_init)["data_x"],verbose=0)[0][self.char2index(pred_init)]
            inf_gain = 1-(-pred_prob*np.log(pred_prob))
            inf_gain_tot = inf_gain_tot + inf_gain
#             print(inf_gain)
            j+=1
#         print(inf_gain_tot)
        try:
            self.inf_gain_prob = inf_gain_tot/name_len
        except:
            self.inf_gain_prob = 1  
            print(self.inf_gain_prob)
        return self.inf_gain_prob
            
    def get_train_accuracy(self,data,threshold):
        self.threshold = threshold
        self.train_data_testing = self.data_loader(data)
        len_name_data = len(self.train_data_testing)
        self.data_prob_list = []
        for i in self.train_data_testing:
#             print(i)
            data_prob = self.get_name_pred_prob(self.train_data_testing[i])
            self.data_prob_list.append(data_prob)
        valid_list = [x for x in self.data_prob_list if x > self.threshold]
        self.train_accuracy = len(valid_list)/len(self.data_prob_list)
        print(self.train_accuracy)
        return self.train_accuracy
    
    def get_train_inf_gain_accuracy(self, data, inf_gain_threshold):
        self.inf_gain_threshold = inf_gain_threshold
        self.train_data_testing_ig = self.data_loader(data)
        len_name_data = len(self.train_data_testing_ig)
        self.inf_gain_prob_list = []
        for i in self.train_data_testing_ig:
#             print(i)
            inf_gain_prob = self.get_name_pred_info_gain_prob(self.train_data_testing_ig[i])
            self.inf_gain_prob_list.append(inf_gain_prob)
        valid_list = [x for x in self.inf_gain_prob_list if x > self.inf_gain_threshold]
        self.train_inf_gain_accuracy = len(valid_list)/len(self.inf_gain_prob_list)
        print(self.train_inf_gain_accuracy)
        return self.train_inf_gain_accuracy
    