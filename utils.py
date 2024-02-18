#!/usr/bin/env python
# coding: utf-8

### 数据加载
import pandas as pd
import numpy as np
import copy
import torch
from common.functions import standardization, return_standardization    # 具体看调用文件的位置

# 直接以的股价为研究对象
# 以 next out_size步收盘价为y，且x归一化
# 输出：
# x:[none,feature_size]
# y_multi: [none,out_size](默认out_size=1）
# X_test, y_multi_test 是测试集数据，基于训练集的 data_max,data_min 标准化的
# 注意：当out_size>1时，前一时间点的y 还是含未知信息的，（当out_size>1时要小心，需要进一步调整。）
def single_data(out_size, data_path, label_loc, test_ratio=0.1):
    
    # out_size 是预测未来的步长
    # label_loc 是目标变量位于data中的列数 （从0数起）
    #  y 是当天的 目标变量的值
    #  y_multi 是未来 out_size 步对应的目标的值
    # data_size_0 是原始数据的大小
    #  data_size 是已考虑未来的out_size 步长的有效单点样本量
    #  feature_size 特征的个数
    #  data_max, data_min 为训练集中每个特征对应的最大最小的值
    data_all = pd.read_csv(data_path).drop(["date"],axis=1)
    X_hat = np.array(data_all)       #np.array要不然后面无法使用 np.的一些属性， .reshape 等  + .drop 删除date变量
    
    data_size_0 = X_hat.shape[0] 
    feature_size = X_hat.shape[1]
    #print(X_hat.shape)
     
    y = copy.copy(X_hat[:, label_loc])   #  y是在第3列上面 (从0起)， 用copy，防止y 随X_hat
    test_size = int(data_size_0*test_ratio)
    
    X_train = X_hat[:-test_size]  # 取前1-ratio部分作为训练集
    y_train = y[:-test_size]
   
    X, data_max, data_min = standardization(X_train)#  将训练（包含train和val）集归一化，所有维度上的数都变成0-1之间的数值  
    
    data_size = (data_size_0-test_size) - out_size   # 训练集包含train和val）data_size 为最大下指标
    X = X[0:data_size]
    y_multi = [ y_train[i:i+data_size].reshape((-1, 1)) for i in range(1, out_size+1) ] # 注意指标，如果用 -，则都用-， 否则乱
    y_multi = np.concatenate(y_multi, axis=1)    # [data_size, out_size]
    
    X_test = X_hat[-test_size:]  # 后ratio部分作为test集
    y_test = y[-test_size:]
    
    ## 类似上面
    X_test,_,_ = standardization(X_test,data_max=data_max,data_min=data_min)  # 测试集标准化（利用训练集的data_max，data_min）
    data_size2 = test_size - out_size  #  data_size2 测试集最大的相对下指标
    
    X_test = X_test[0:data_size2]
    y_multi_test = [ y_test[i:i+data_size2].reshape((-1, 1)) for i in range(1, out_size+1) ] # 注意指标，如果用 -，则都用-， 否则乱
    y_multi_test = np.concatenate(y_multi_test, axis=1)
    
    return X, y_multi, data_size, feature_size, X_test, y_multi_test, data_max, data_min

# 以的股票收益率为研究对象
# 以 next out_size步收益率为y，且x归一化
# 输出：
# x:[none,feature_size] + 当前股票收益率，用于二次建模AR的滞后ut
# y_multi: [none,out_size](默认out_size=1, 
# X_test, y_multi_test 是测试集数据，基于训练集的 data_max,data_min 标准化的
# 注意：当out_size>1时，前一时间点的y 还是含未知信息的，（当out_size>1时要小心，需要进一步调整。）
def single_data_r(out_size, data_path, label_loc, test_ratio=0.1):
    
    # 以 next 对数收益率为y，且x归一化 
    # out_size 是预测未来的步长
    # label_loc 是目标变量位于data中的列数 （从0数起）
    #  data_size 是已考虑收益率的未来的out_size 步长的有效单点样本量
    #  feature_size 特征的个数
    #  data_max, data_min 为训练集中每个特征对应的最大最小的值
    X, y_multi, data_size1, feature_size, X_test, y_multi_test, data_max, data_min = single_data(out_size, data_path, label_loc,test_ratio=test_ratio )
                                                                  # y_multi：[none,out_size] 
    X_close = return_standardization(data = X, data_max=data_max, data_min=data_min, label_loc=label_loc)  # 对目标变量反标准化
    X_ret = 100 * np.log(X_close[1:] / X_close[0:-1])     # 当前的收盘价的对数收益率
    y_multi = 100 * np.log(y_multi[1:] / y_multi[0:-1])   # 未来的out_size步长的对数收益率
    X = np.concatenate([X[1:],X_ret.reshape(-1,1)],axis=1) # 将当前的收盘价的对数收益率存入X中
    data_size = data_size1 - 1

    X_test_close = return_standardization(data = X_test, data_max=data_max, data_min=data_min, label_loc=label_loc)  # 对目标变量反标准化
    X_test_ret = 100 * np.log(X_test_close[1:] / X_test_close[0:-1]) 
    y_multi_test = 100 * np.log(y_multi_test[1:] / y_multi_test[0:-1])   # 未来的out_size步长的对数收益率
    X_test = np.concatenate([X_test[1:],X_test_ret.reshape(-1,1)],axis=1) # 将当前的收盘价的对数收益率存入X中

    return X, y_multi, data_size, feature_size, X_test, y_multi_test, data_max, data_min


##
# 以股票波动率为研究对象
# 以 next out_size步波动为y，且x归一化
# 输出：
# x:[none,feature_size] + 当前股票收益率 + 当前股票波动率，用于二次建模AR的滞后ut
# y_multi_v: [none,out_size](默认out_size=1, 
# X_test, y_multi_test_v 是测试集数据，基于训练集的 data_max,data_min 标准化的
# 注意：当out_size>1时，前一时间点的y 还是含未知信息的，（当out_size>1时要小心，需要进一步调整。）
def single_data_v(out_size, data_path, label_loc, test_ratio=0.1, val_len=22):
    
    X, y_multi, data_size1, feature_size, X_test, y_multi_test, data_max, data_min = single_data_r(out_size, data_path,\
                                                             label_loc, test_ratio=test_ratio)
    
    data_size = data_size1 - val_len + 1   # sample size    ?-out_size     
    X_v = [X[i:i+val_len,-1].reshape(-1,1).var(axis=0, keepdims=True) for i in range(data_size) ]   # 当前收益率的波动率
    X_v = np.concatenate(X_v, axis=0)
    X = np.concatenate([X[val_len-1:],X_v.reshape(-1,1)],axis=1) # 将当当前收益率的波动率存入X中
    y_multi_v = [y_multi[i:i+val_len].var(axis=0, keepdims=True) for i in range(data_size) ] # 未来的out_size步长的波动率
    y_multi_v = np.concatenate(y_multi_v, axis=0)


    data_size_test = X_test.shape[0] - val_len + 1    #  val_len 
    X_test_v = [X_test[i:i+val_len,-1].reshape(-1,1).var(axis=0, keepdims=True) for i in range(data_size_test) ]   # 当前收益率的波动率
    X_test_v = np.concatenate(X_test_v, axis=0)
    X_test = np.concatenate([X_test[val_len-1:],X_test_v.reshape(-1,1)],axis=1) # 将当前收益率的波动率存入X中
    y_multi_test_v = [y_multi_test[i:i+val_len].var(axis=0, keepdims=True) for i in range(data_size_test) ] 
    y_multi_test_v = np.concatenate(y_multi_test_v, axis=0)
     
    return X, y_multi_v, data_size, feature_size, X_test, y_multi_test_v, data_max, data_min


# single_data_d 只考虑一个时间点的真值，一阶二阶差分 
# 股价或收益率的差分样本
# 输出：
# x2: [none,feature_size]
# y_multi2: [none, 1 + difference_order，out_size](默认out_size=1 ) or [none, outsize](difference_order=None)
# X_test2, y_multi_test2 是测试集数据，基于训练集的 data_max,data_min 标准化的
# 注意：当out_size>1时，前一时间点的y 还是含未知信息的
def single_data_d(out_size, data_path, label_loc, difference_order=2, pattern='price',test_ratio=0.1):# 以 next 收盘价为y，且x归一化
    
    # out_size 是预测未来的步长
    # label_loc 是目标变量位于data中的列数 （从0数起） 
    # data_size1 是已考虑未来的out_size 步长的有效单点样本量
    # data_size2 是进一步差分之后的样本量的大小 
    # data_max, data_min 为训练集中每个特征对应的最大最小的值
    # y_multi 是未来 out_size 步对应的目标的值，y_multi: [None, out_size]
    # y_multi2: [None,1+ difference_order,out_size]
    
    if pattern=='price':              # 股价原来的out_size步长样本, data_size1为单点样本量
        X, y_multi,data_size1,feature_size,X_test, y_multi_test,data_max,data_min = \
         single_data(out_size, data_path, label_loc,test_ratio=test_ratio) 
                                  
    elif pattern=='yield':             # 收益率原来的out_size步长样本, data_size1为单点样本量
        X, y_multi, data_size1, feature_size, X_test, y_multi_test,data_max,data_min\
        = single_data_r(out_size, data_path, label_loc,test_ratio=test_ratio)   
    else:
        raise ValueError('only "price" or "yield" ')
    
    if difference_order==None:
        return X, y_multi,data_size1,feature_size,X_test, y_multi_test,data_max,data_min 
    
    data_size2 = data_size1 - difference_order   # 差分之后的样本量的大小
    X2 = X[difference_order:]           # difference_order是差分要求的最小样本起始位置   # 预测一二差分，可能用到未知信息？？？？？？？ 
    
    y_multi1 = y_multi
    y_multi2 = []
    for i in range(difference_order):   
        y_multi1 = y_multi1[1:]- y_multi1[0:-1]  # 逐次做一阶差分（做difference_order 次）
        y_multi2.append( np.expand_dims(y_multi1[difference_order-1-i:], axis=1) ) # 不同阶差分的起始位置
    y_multi2 = np.concatenate(y_multi2, axis=1)
    
    y_multi2 = np.concatenate([np.expand_dims(y_multi[difference_order:],axis=1), y_multi2 ], axis=1)  # 并上y的值
    
    ################### 同上操作X_test，y_multi_test
    X_test2 = X_test[difference_order:]
    y_multi_test1 = y_multi_test
    y_multi_test2 = []
    for i in range(difference_order):   
        y_multi_test1 = y_multi_test1[1:]- y_multi_test1[0:-1]  # 逐次做一阶差分（做difference_order 次）
        y_multi_test2.append( np.expand_dims(y_multi_test1[difference_order-1-i:], axis=1) ) # 不同阶差分的起始位置
    y_multi_test2 = np.concatenate(y_multi_test2, axis=1)
    
    y_multi_test2 = np.concatenate([np.expand_dims(y_multi_test[difference_order:],axis=1), y_multi_test2 ], axis=1)  # 并上y的值
    ###########################
    
    return X2, y_multi2, data_size2, feature_size, X_test2, y_multi_test2,data_max,data_min   

##############################################################################################
#### 随机抽取训练和验证集起始index集
# --> array, array, array  分别是训练、验证、测试起始指标的集合 

# def train_val_test(data_size, seq_len, train_ratio, val_ratio):
        
#     # seq_len 为 window_size, max_index 为抽样起始指标的最大值
#     # train_size  训练集大小
#     # val_size  验证集大小
#     # test_size  测试集大小
#     # test_index 测试集起始指标的集合（分割的最后一整部分为测试集）
#     # train_val_index 训练集与验证集起始指标的集合，（前面部分为 训练集和验证集）
#     # train_index 是随机抽取train set 对应的起始指标的集合
#     # val_index 随机剩下的为 验证集起始指标的集合
#     # max_index 是seq_len序列中最大的起始位置，（也是seq_len序列的样本的 sample size ）
    
#     max_index = data_size - seq_len  
#     train_size = int(max_index * train_ratio)  
#     val_size = int(max_index * val_ratio)   
#     test_size = max_index - train_size - val_size  
#     test_index = range(train_size + val_size, max_index)
#     train_val_index = range(train_size + val_size)
#     train_index = np.random.choice(train_val_index, train_size, replace=False, p=None)  
#     val_index = np.array(list(set(train_val_index)- set(train_index)))
    
#     return train_index, val_index, test_index  

def train_val_test(data_size, seq_len, train_ratio):
        
    # seq_len 为 window_size, max_index 为抽样起始指标的最大值
    # train_size  训练集大小
    # train_val_index 训练集与验证集起始指标的集合
    # train_index 是随机抽取train set 对应的起始指标的集合
    # val_index 随机剩下的为 验证集起始指标的集合
    # max_index 是seq_len序列中最大的起始位置，（也是seq_len序列的样本的 sample size ）
    
    max_index = data_size - seq_len 
    train_val_index = range(max_index)
    
    train_size = int(max_index * train_ratio) 
    train_index = np.random.choice(train_val_index, train_size, replace=False, p=None)  
    val_index = np.array(list(set(train_val_index)- set(train_index)))
    
    return train_index, val_index  

### 以下编写按顺序抽取样本
# def train_val_test0(data_size, seq_len, train_ratio, val_ratio):
        
#     # seq_len 为 window_size, max_index 为抽样起始指标的最大值
#     # train_size  训练集大小
#     # val_size  验证集大小
#     # test_size  测试集大小
#     # test_index 测试集起始指标的集合（分割的最后一整部分为测试集）
#     # train_val_index 训练集与验证集起始指标的集合，（前面部分为 训练集和验证集）
#     # train_index 是随机抽取train set 对应的起始指标的集合
#     # val_index 随机剩下的为 验证集起始指标的集合
#     # max_index 是seq_len序列中最大的起始位置，（也是seq_len序列的样本的 sample size ）
    
#     max_index = data_size - seq_len  
#     train_size = int(max_index * train_ratio) 
#     train_index = range(train_size)
#     val_size = int(max_index * val_ratio)  
#     val_index = range(train_size,train_size+val_size)
#     test_size = max_index - train_size - val_size  
#     test_index = range(train_size + val_size, max_index) 
#     return train_index, val_index, test_index   

def train_val_test0(data_size, seq_len, train_ratio):
        
    # seq_len 为 window_size, max_index 为抽样起始指标的最大值
    # train_size  训练集大小
    # test_index 测试集起始指标的集合（分割的最后一整部分为测试集）
    # train_val_index 训练集与验证集起始指标的集合，（前面部分为 训练集和验证集）
    # train_index 是随机抽取train set 对应的起始指标的集合
    # val_index 随机剩下的为 验证集起始指标的集合
    # max_index 是seq_len序列中最大的起始位置，（也是seq_len序列的样本的 sample size ）
    
    max_index = data_size - seq_len  
    train_size = int(max_index * train_ratio) 
    train_index = range(train_size)
    val_index = range(train_size,max_index)
   
    return train_index, val_index   

#### 根据起始指标集进行抽样
# --> tensor  batch_X:[seq_len, batch, feature],  batch_Y: [batch, seq, 1+difference_oder，outsize] or [batch, seq, outsize](difference_order=None)
def batch_sample(X, Y, batch_index, seq_len): # 配合single_data_d使用
    
    # batch_index 是起始指标的集
    # seq_len 是window大小
    # batch_Xt 转化为 tensor  的 batch sample  
    # batch_Yt 转化为 tensor  的 batch sample
    # batch_end_index 为 batch 最大下角标
    batch_X, batch_Y = [], []
    batch_X = [ np.expand_dims(X[index : index + seq_len], 0) for index in batch_index ]
    batch_Y = [ np.expand_dims(Y[index : index + seq_len], 0) for index in batch_index ]
    batch_X = np.concatenate(batch_X, axis=0).transpose(1, 0, 2)  #  [seq_len, batch, feature+2]
    batch_Y = np.concatenate(batch_Y, axis=0)              #  [batch, seq, outsize]
    batch_Xt = torch.from_numpy(batch_X).type(torch.float32)
    batch_Yt = torch.from_numpy(batch_Y).type(torch.float32) 
    
    return batch_Xt, batch_Yt  

####
#  给定 all_X, all_Y, all_batch_index  根据 batch_sample 抽出 b_all_x,  b_all_y  
def all_batch_sample(all_X, all_Y, all_batch_index, seq_len): # all_X list: all_y:list,all_batch_index:list , seq_len : int
    b_all_x, b_all_y = [],[]
    for X, y_multi, batch_index in zip(all_X, all_Y, all_batch_index):      
            b_x, b_y = batch_sample(X, y_multi, batch_index, seq_len)
            #b_x, b_y =b_x.to(device), b_y.to(device)  # 数据转至 device (GPU)
            b_all_x.append(b_x)
            b_all_y.append(b_y)  # b_all_x list: [b_x1,b_x2...], b_x1 ;[seq_len，batch，feature]
    return b_all_x, b_all_y     # b_all_y list: [b_y1,b_y2,..], b_y1:[batch, out_size]           # 数据转至device (GPU)





