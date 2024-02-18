#!/usr/bin/env python
# coding: utf-8

### 数据标准化
import numpy as np

# 标准化数据（包括test集上的标准化）
def standardization(data, data_max=None, data_min=None):  # min，max标准化 平移中心点 
    try:
        if data_max==None:
            data_max = np.max(data, axis=0, keepdims=True)
            data_min = np.min(data, axis=0, keepdims=True) 
    except:
        pass
    #return (data - data_min)/(data_max - data_min), data_max, data_min
    return (data - data_min)/(data_max - data_min)-1/2, data_max, data_min   #  -1/2

def return_standardization(data,data_max,data_min,label_loc=None):
    if label_loc==None:
        X = (data+0.5)* (data_max-data_min) + data_min
    else:
        X = (data[:,label_loc]+0.5)* (data_max-data_min)[0,label_loc] + data_min[0,label_loc]
        X = X.reshape(-1,1)  ## 保持维度 
    return X
    
def standardization1(data,data_max=None,data_min=None): # min，max标准化
    try:
        if data_max==None:
            data_max = np.max(data, axis=0,keepdims=True)
            data_min = np.min(data, axis=0,keepdims=True) 
    except:
        pass
    return (data - data_min)/(data_max - data_min), data_max, data_min 

def standardization2(data,mu=None,sigma=None): # 标准正态化，但金融数据一般是后尾数据，故常采用上述 min，max标准化
    try:
        if mu==None:
            mu = np.mean(data, axis=0,keepdims=True)
            sigma = np.std(data, axis=0,keepdims=True)
    except:
        pass
    return (data - mu) / sigma, mu, sigma               # 需要添加稳定因子 eps=0.0001?