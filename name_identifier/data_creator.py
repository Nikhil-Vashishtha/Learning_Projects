# importing modules

import os
import sys
import numpy as np
import pandas as pd
import random
import string
import pickle
import bz2
from keras.utils import np_utils

class data_creator_class():
    def __init__(self,split_ratio):
        self.split_ratio = split_ratio
        work_path = "D:/Chrome_downloads/name_identifier/"
        data_path = work_path + "resources/"
        self.save_data_path = work_path + "data/"
        self.total_data = pd.read_csv(data_path+"names.csv")
        self.total_data.shape
        self.total_data.columns
        self.total_data.name = self.total_data.name.astype(str)
        self.names_array = np.array(self.total_data["name"].values.tolist())
        
    def preproc_names(self, name):
        name_ = "".join([i for i in name if i not in string.punctuation])
        name_ = name_.lower()
        
        return name_
    
    def create_names_matrix(self, x):
        names_matrix = np.zeros((self.max_name_len_ext,27))
        for i in range(len(x)):
            try:
                names_matrix[i][string.ascii_lowercase.index(x[i])]=1
            except:
                names_matrix[i][-1]=1
                
        return names_matrix
    
    def split_data(self):
        self.names_array_v2=np.array([self.preproc_names(z) for z in self.names_array])
        self.max_name_len = max([len(x) for x in self.names_array_v2])
        self.max_name_len_ext = max([len(x) for x in self.names_array_v2])+10
        names_indexes=list(range(len(self.names_array_v2)))
        names_dict = {}
        names_matrix_dict = {}
        for i in names_indexes:
            names_dict[i]=self.names_array_v2[i]
            names_matrix_dict[i]=self.create_names_matrix(self.names_array_v2[i])
        self.split_data_dict={}
        train_len = int(len(names_dict)*self.split_ratio)
        print(train_len)
        train_indices = random.sample(range(0, len(names_dict)), train_len)
        test_indices = list(set(list(range(len(names_dict)))) -set(train_indices))
        self.split_data_dict["train_indices"]=train_indices
        self.split_data_dict["test_indices"]=test_indices
        self.train_names_dict = {}
        self.test_names_dict = {}
        self.train_data_dict = {}
        self.test_data_dict = {}
        for i in self.split_data_dict["train_indices"]:
            self.train_data_dict[i] = names_matrix_dict[i]
            self.train_names_dict[i] = names_dict[i]
        for j in self.split_data_dict["test_indices"]:
            self.test_data_dict[j] = names_matrix_dict[j]
            self.test_names_dict[j] = names_dict[j]
            
        return None
    
    def create_data(self,slide_kernel):
        self.slide_kernel = slide_kernel
        dataX = []
        dataY = []
        for j in self.train_data_dict:
            for i in range(0, self.max_name_len_ext - self.slide_kernel, 1):
                seq_in = self.train_data_dict[j][i:i + self.slide_kernel]
                if np.count_nonzero(seq_in)!=0:
                    seq_out = self.train_data_dict[j][i + self.slide_kernel]
                    dataX.append(seq_in)
                    dataY.append(seq_out)
                else:
                    pass
        dataX_test = []
        dataY_test = []
        for j in self.test_data_dict:
            for i in range(0, self.max_name_len_ext - self.slide_kernel, 1):
                seq_in_test = self.test_data_dict[j][i:i + self.slide_kernel]
                if np.count_nonzero(seq_in_test)!=0:
                    seq_out_test = self.test_data_dict[j][i + self.slide_kernel]
                    dataX_test.append(seq_in_test)
                    dataY_test.append(seq_out_test)
                else:
                    pass 
        
        assert len(dataX) == len(dataY)
        assert len(dataX_test) == len(dataY_test)
        
        n_patterns = len(dataX)
        print(n_patterns)
        
        n_patterns_test = len(dataX_test)
        print(n_patterns_test)

        self.dataX_arr = np.array(dataX)
        print(self.dataX_arr.shape)

        self.dataY_arr = np.array(dataY)
        print(self.dataY_arr.shape)
        
        self.dataX_test_arr = np.array(dataX_test)
        print(self.dataX_test_arr.shape)

        self.dataY_test_arr = np.array(dataY_test)
        print(self.dataY_test_arr.shape)
        
        return None
        
    def save_data(self,data,name):
        save_handle = bz2.BZ2File(self.save_data_path + name, 'wb')
        pickle.dump(data, save_handle)
        save_handle.close()
        
    def data_saver_all(self):
        xtrain_savename = "dataX_train_krnl_"+str(self.slide_kernel)+".bz2"
        self.save_data(self.dataX_arr,xtrain_savename)
        print(xtrain_savename)
        
        ytrain_savename = "dataY_train_krnl_"+str(self.slide_kernel)+".bz2"
        self.save_data(self.dataY_arr,ytrain_savename)
        print(ytrain_savename)
        
        xtest_savename = "dataX_test_krnl_"+str(self.slide_kernel)+".bz2"
        self.save_data(self.dataX_test_arr,xtest_savename)
        print(xtest_savename)
        
        ytest_savename = "dataY_test_krnl_"+str(self.slide_kernel)+".bz2"
        self.save_data(self.dataY_test_arr,ytest_savename)
        print(ytest_savename)

        train_names_dict_name = "train_names_dict_krnl_"+str(self.slide_kernel)+".bz2"
        self.save_data(self.train_names_dict,train_names_dict_name)
        print(train_names_dict_name)
        
        train_data_dict_name = "train_data_dict_krnl_"+str(self.slide_kernel)+".bz2"
        self.save_data(self.train_data_dict,train_data_dict_name)
        print(train_data_dict_name)
        
        test_names_dict_name = "test_names_dict_krnl_"+str(self.slide_kernel)+".bz2"
        self.save_data(self.test_names_dict,test_names_dict_name)
        print(test_names_dict_name)
        
        test_data_dict_name = "test_data_dict_krnl_"+str(self.slide_kernel)+".bz2"
        self.save_data(self.test_data_dict,test_data_dict_name)
        print(test_data_dict_name)
        
        return None
