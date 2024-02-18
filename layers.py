#!/usr/bin/env python
# coding: utf-8

### 位置编码


import numpy as np
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=30, dropout_p=0.1):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)    # 20*512
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)    # 20*1  
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 对照公式   
        pe[:, 0::2] = torch.sin(position * div_term)             #  math.log() 也可使用 torch.log( tensor ) 
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)        # transformerencoder  默认是 seq  batch  d_model     
        self.register_buffer('pe', pe)   #20*1*512   # 模型中常量值，加载保存的时候,自带模型的buffer
        self.dropout = nn.Dropout(dropout_p)
  
    def forward(self, x):     # 这里的 x 要求 [seq,batch,d_model]  (seq 包括 RG_token)
        return self.dropout(x + self.pe[:x.size(0),:])  #20*8*512   广播效应
