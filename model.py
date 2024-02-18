
import torch
import torch.nn as nn
from einops import rearrange, repeat
from common.layers import PositionalEncoding
import numpy as np   

##### 该块代码是基于YH/Volatility的 modelnew 的基础上修改并分块的， 该部分是多源数据的融入方式（写于2023_1113）

##############该模型model与 modelnew 区别就是用类的多态性减少建模里的if 判断语句。
##########################################

# 主要是三个方面的选择 
# 1 嵌入方式（如何嵌入效果较佳）  
#嵌入的模式 ：  9种
## "clspric" ：仅收盘价嵌入
## "numerical_no_onehot"：仅数值整块嵌入，不含独热编码信息
## "numerical"：数值整块嵌入 + 独热编码信息
## "numerical2"：数值分2块嵌入 + 独热编码信息
## "total"：全部信息整体嵌入
## "classify"：信息分类嵌入，再求和 
## "separately"：信息分别嵌入，再求和
## "mix"：混合嵌入，提取出收盘价开盘价（数值分2整块嵌入)，再求和
## "text"：仅仅嵌入文本

# 2 提取特征（RG 和 全局avg 池化） 
# 3 构建模型 （直接建模"None"，二次建模选有"garch"、"tgarch","egarch"）

########## 嵌入的模式
'''
feature (81) : fea1, fea2, fea3, fea4 = 9, 15, 50, 5  分别为 股价、技术、文本、星期(79) |  2 当前时刻收益率和波动率，用于二次建模

d_model：每种信息嵌入后的维数
numerical_num：除独热编码外的 数值的特征变量个数  24=9+15
word_vector_dim：词嵌入的维数  50
onehot_index：独热编码所在的位置，默认放在数据的最后7-2列，range(-7,-2)
在父类Embedding 中，先设计好默认参数便于子类调用

注意：这里增加了独热编码了，在嵌入的时候要注意。
'''

class Embedding(nn.Module):   ## 验证后发现，需要添加父类，若不然Embedding层无法加入model进行更新          
    def __init__(self, d_model, numerical_num=24, onehot_index=range(-7,-2), word_vector_dim=50):
        nn.Module.__init__(self)
        self.d_model = d_model       
        self.numerical_num = numerical_num  
        self.word_vector_dim = word_vector_dim    
        self.onehot_index = onehot_index         
    def embedding(self):
        return None

# 仅收盘价嵌入
class clspric(Embedding):
    def __init__(self, d_model):
        Embedding.__init__(self,d_model)
        input_dim = 1
        self.label_loc = 0  ## 收盘价loc
        self.embedding_layer = nn.Linear(input_dim, self.d_model)    
    def embedding(self,src):
        src_out = self.embedding_layer( torch.unsqueeze(src[:, :, self.label_loc], -1) )
        return src_out
    
# 数值整块嵌入，不含独热编码信息
class numerical_no_onehot(Embedding):
    def __init__(self, d_model):
        Embedding.__init__(self,d_model)
        input_dim = self.numerical_num
        self.embedding_layer = nn.Linear(input_dim, self.d_model)  
    def embedding(self,src):
        src_out = self.embedding_layer(src[:, :, range(self.numerical_num)])
        return src_out

# 数值整块嵌入 + 独热编码信息   
class numerical(Embedding):
    def __init__(self, d_model):
        Embedding.__init__(self,d_model)
        input_dim = self.numerical_num + len(self.onehot_index)
        self.embedding_layer = nn.Linear(input_dim, self.d_model)
    def embedding(self,src):
        index = np.r_[range(self.numerical_num), self.onehot_index]  
        src_out = self.embedding_layer(src[:, :, index])
        return src_out
    
# 数值分2块嵌入 + 独热编码信息   
class numerical2(Embedding):
    def __init__(self, d_model):
        Embedding.__init__(self,d_model)  
        self.input_dim1 = 9 + 5  # 股价9  + 星期独热编码 5
        self.input_dim2 = 15  # 技术15 
        self.embedding_layer = nn.ModuleList([nn.Linear(self.input_dim1, self.d_model),\
                                nn.Linear(self.input_dim2, self.d_model)])        

    def embedding(self,src):
        index = np.r_[range(9), self.onehot_index]   
        src_out0 = self.embedding_layer[0](src[:, :, index])
        src_out1 = self.embedding_layer[1](src[:, :, range(9, 9+self.input_dim2)])
        src_out = src_out0 + src_out1
        return src_out

# 全部信息整体嵌入
class total(Embedding):
    def __init__(self, d_model):
        Embedding.__init__(self,d_model) 
        self.input_dim = self.numerical_num + self.word_vector_dim + len(self.onehot_index)
        self.embedding_layer = nn.Linear(self.input_dim, self.d_model) 
    def embedding(self, src):
        src_out = self.embedding_layer(src[:, :, range(self.input_dim)])   
        return src_out


# 信息分类嵌入，再求和  
class classify(Embedding):
    def __init__(self, d_model):
        Embedding.__init__(self,d_model)  
        self.input_dim1 = 9 + 15 + 5  # 数值数据 ： 股价9  +  技术15 + 星期独热编码 5
        self.input_dim2 = 50 # 文本数据 ：50 
        self.embedding_layer = nn.ModuleList([nn.Linear(self.input_dim1, self.d_model),\
                                nn.Linear(self.input_dim2, self.d_model)])        

    def embedding(self,src):
        index = np.r_[range(24), self.onehot_index]   
        src_out0 = self.embedding_layer[0](src[:, :, index])
        src_out1 = self.embedding_layer[1](src[:, :, range(24, 24 + self.input_dim2)])
        src_out = src_out0 + src_out1
        return src_out
    
    
# 信息分别嵌入，再求和  
class separately(Embedding):
    def __init__(self, d_model):
        Embedding.__init__(self,d_model) 
        self.input_dims = []
        self.input_dims.append(9)     #  原始数据
        self.input_dims.append(15)    #  技术指标
        self.input_dims.append(self.word_vector_dim)   #  研报
        self.input_dims.append(len(self.onehot_index))     # 独热编码
        
        self.embedding_layer = []
        for input_dim in self.input_dims:
            self.embedding_layer.append(nn.Linear(input_dim, self.d_model))
        self.embedding_layer = nn.ModuleList(self.embedding_layer)
            
    def embedding(self, src):
        src_outs = []
        start_index = 0
        for i, sub_embedding in enumerate(self.embedding_layer):
            src_outs.append(sub_embedding(src[:, :, range(start_index, start_index + self.input_dims[i] ) ] ) ) 
            start_index += self.input_dims[i]   
        src_out = sum(src_outs)   # 对对象元素求和
        return src_out
    
    
# 混合嵌入，提取出收盘价开盘价（数值分2整块嵌入)，再求和
class mix(Embedding):
    def __init__(self, d_model):
        Embedding.__init__(self,d_model)  
        self.input_dim1 = 2 + 5  # 收盘价开盘价 2  + 星期独热编码 5  ？？？
        self.input_dim2 = 22  # 其他股价数据 7 + 技术指标 15 
        self.input_dim3 = self.word_vector_dim
        self.embedding_layer = nn.ModuleList([nn.Linear(self.input_dim1, self.d_model),\
                                nn.Linear(self.input_dim2, self.d_model),\
                                nn.Linear(self.input_dim3, self.d_model)])        
    def embedding(self,src):
        index = np.r_[range(2), self.onehot_index]   
        src_out0 = self.embedding_layer[0](src[:, :, index])
        src_out1 = self.embedding_layer[1](src[:, :, range(2, 2+self.input_dim2)])
        src_out2 = self.embedding_layer[2](src[:, :, range(2+self.input_dim2, 2+self.input_dim2+self.input_dim3)])
        src_out = src_out0 + src_out1 + src_out2
        return src_out    
            
        
# 仅仅嵌入文本
class text(Embedding):
    def __init__(self, d_model):
        Embedding.__init__(self,d_model) 
        self.input_dim = self.word_vector_dim
        self.embedding_layer = nn.Linear(self.input_dim, self.d_model) 
    def embedding(self, src):
        src_out = self.embedding_layer(src[:, :, range(24, 24+self.input_dim)])   
        return src_out                                     
                               
                               
def trans_layers_paramters(num_layers):   # transformer d_model, nhead, dim_feedforward
    if num_layers==3:
        d_model=192; nhead=6
    elif num_layers==4:
        d_model=384; nhead=6
    elif num_layers==6:
        d_model=512; nhead=8
    elif num_layers==8:
        d_model=576; nhead=9
    elif num_layers==12:
        d_model=768; nhead=12
    else:
        raise ValueError
    dim_feedforward = 4 * d_model
    return d_model, nhead, dim_feedforward
    

class TransAm(nn.Module): # [seq,batch,feature_size+2] --> [batch,out_size] or [batch,(1+2*ar_p+ma_q)*out_size] or  [batch,(1+3*ar_p+ma_q)*out_size]  
    def __init__(self, out_size, d_model=512, nhead=8, dim_feedforward=2048, num_layers=6, dropout=0.5, \
                 pool="RG", embedding_mode='separately', ar_p = 1, ma_q = 1, garch_mode="None"):      
        # out_sie :预测步长
        # embedding_mode :嵌入模式共9钟
        # pool = 'RG' or 'mean'  :# 分别对应最后一层 回归是基于 RG位置，还是全局特征平均
        # ar_p、ma_q: garch类建模滞后项数
        # garch_mode: "None" , "garch" , "tgarch", "egarch"

        super(TransAm, self).__init__()
        
        self.model_type = 'Transformer'
        self.src_mask = None
             
        self.RG_token = nn.Parameter(torch.randn(1, 1, d_model))  #### nn.Parameter()定义可学习参数#新加的东西

        self.embedding_layer= eval(embedding_mode)(d_model)   
        
        self.relu = nn.ReLU()
                                        
        self.pos_encoder = PositionalEncoding(d_model) # 50*512
       
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,\
                                        dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
            
        self.init_transformer_in_decoder(out_size, d_model, garch_mode, ar_p, ma_q)   # 设计 transfomer 内部的decoder
        
        self.init_weights()  # 初始化自定义的参数
        self.src_key_padding_mask = None
        # self.to_latent = nn.Identity()
        self.pool = pool
        self.embedding_mode = embedding_mode
        self.label_loc = 0    #  目标变量在数据中对应的位置
        self.d_model = d_model
        self.out_size = out_size
       
    def init_transformer_in_decoder(self,out_size, d_model, garch_mode, ar_p, ma_q):   # 设计 transfomer 内部的decoder
        if garch_mode == "None": ## 直接预测 sigma2
            self.decoder = nn.Linear(d_model,out_size)  # 原始以直接的目标为预测对象的解码块
        elif garch_mode == "garch":
            self.decoder = nn.Linear(d_model, (ar_p+1+ar_p+ma_q) * out_size) ## 滞后条件均值ar_p+常数项1+ar_p系数+ma_q系数
        elif garch_mode == "tgarch":
            self.decoder = nn.Linear(d_model, (ar_p+1+ar_p+ma_q+ar_p) * out_size) ## 滞后条件均值ar_p+常数项1+ar_p系数+ma_q系数+ar_p杠杆的系数
        elif garch_mode == "egarch":
            self.decoder = nn.Linear(d_model, (ar_p+1+ar_p+ma_q+ar_p) * out_size) ## 滞后条件均值ar_p+常数项1+ar_p系数+ma_q系数+ar_p杠杆的系数
        else:
            print( 'garch_mode must one of "None","garch","tgarch","egarch"')  
            raise ValueError
        return None

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_padding): #  shape of src  [seq,batch,feature_size] --> [batch,out_size]      
                                                  
        if self.src_key_padding_mask is None:
            mask_key = src_padding               #[batch,seq]
            self.src_key_padding_mask = mask_key
          
        # 对多源数据嵌入方式的选择
        src = self.embedding_layer.embedding(src)   # __init__中已实例化                  
                               
        # 若 self.pool == "RG"，则添加一个回归元素位置       
        RG_tokens = repeat(self.RG_token,'s () d -> s b d', b=src.shape[1])  # [seq, batch, d_model]  
        if self.pool == "RG": src = torch.cat((RG_tokens, src), dim=0 )       
        
        # 对数据源信息src添加位置编码
        src = self.pos_encoder(src) # [seq,batch,d_model]  
        output_encoder = self.transformer_encoder(src, self.src_mask, self.src_key_padding_mask) #output_encoder:[seq,batch,d_model]
        
        # 提取最终特征的方式 self.pool == "RG"，则用回归元素位置提取特征，否则采用全局平均池化
        input_decoder = output_encoder[0]  if  self.pool == "RG" else output_encoder.mean(dim=0) #  [batch, d_model]   
        #output = self.to_latent(output)
        
        # 解码部分，直接股票波动率，或者 garch类参数，具体很具后续的损失函数定义而定。
        output_decoder = self.decoder(input_decoder)  
        # [batch,out_size] or [batch,(1+2*ar_p+ma_q)*out_size] or  [batch,(1+3*ar_p+ma_q)*out_size](earch和tgarch)
        
        self.src_key_padding_mask = None
        
        return output_decoder

    
## 二次构建garch类模型的部分    
def all_hat_predict(output, out_size ,garch_mode, b_x, ar_p, ma_q):
    
    ########### 模型提取特征的选择######################
    # 上述output对应以下三种情形 
    # 第一种情形output对应着 [None,out_size] 是直接预测真值的。
    # 第二种情形output对应着 [None, 1+2*ar_p+ma_q, out_size]，【garch】, 滞后条件均值ar_p+常数项1+ar_p系数+ma_q系数
    # 第三种情形output对应着 [None, 1+3*ar_p+ma_q, out_size]，【tgarch】,滞后条件均值ar_p+常数项1+ar_p系数+ma_q系数+ar_p杠杆的系数
    # 第四种情形output对应着 [None, 1+3*ar_p+ma_q, out_size]，【egarch】,滞后条件均值ar_p+常数项1+ar_p系数+ma_q系数+ar_p杠杆的系数
    
    output = output.view(output.shape[0], -1, out_size)    # 对output重新排列 [batch,-1,out_size]
    # 对第一种情形直接预测真值
    if garch_mode=="None":               
        y_hat = output    # [batch,1,out_size]
        return y_hat  
    
    # 以下对其他三种情况提取出来的特征进行garch建模预测
    #### 0 先对b_x 提取出 garch类公式需要滞后收益率和滞后波动率   
    b_x_r = [b_x[-i-1,:,-2].reshape(-1,1) for i in range(ar_p)] ## -i-1：滞后一阶...滞后ar_p； -2：收益率
    b_x_r = torch.cat(b_x_r, dim=1)
    b_x_v = [b_x[-i-1,:,-1].reshape(-1,1) for i in range(ma_q)] ## -i-1：滞后一阶...滞后ar_p； -1：波动率
    b_x_v = torch.cat(b_x_v, dim=1)
    b_xs = torch.cat([b_x_r,b_x_v], dim=1)  # [batch, ar_p + ma_q] ar_p阶收益率滞后； ma_q阶波动率滞后
    #b_xs = torch.from_numpy(b_xs)

    #### 1 利用 output系数 和 b_xs滞后真值 进行garch建模   
    ut_hat_p, alpha_0, alpha_p, beta_q ,ret_p, sigma2_q, lever_p = get_Coeff_func(output, b_xs ,ar_p, ma_q) ## lever_p garch用不到
    
    ut_hat_p[1:] = ut_hat_p[:-1].clone()  ## ut_hat_p:[batch, ar_p+ma_q, 1]  ;错开一位 将ut+1 改为 ut   ！！！！ 考虑剩余batch = 1 的情况
    
    if garch_mode=="garch":
        et2 =  (ret_p - ut_hat_p)**2  ## 计算差值平方
        y_hat = alpha_0 + torch.sum(alpha_p*et2,dim=1).unsqueeze(dim=1) + torch.sum(beta_q*sigma2_q,dim=1).unsqueeze(dim=1)
    elif garch_mode=="tgarch":
        et = ret_p - ut_hat_p
        zero = torch.zeros_like(et);one = torch.ones_like(et)
        I_et = torch.where(et>0,zero,et)
        I_et = torch.where(I_et<0,one,I_et)  ## 示性函数 et<0 为1
        et2 = et**2  ## 计算差值平方
        y_hat = alpha_0 + torch.sum(alpha_p*et2,dim=1).unsqueeze(dim=1) + torch.sum(beta_q*sigma2_q,dim=1).unsqueeze(dim=1) \
              + torch.sum(lever_p*I_et*et2, dim=1).unsqueeze(dim=1)
    elif garch_mode=="egarch":
        et = ret_p - ut_hat_p
        zero = torch.zeros_like(et);one = torch.ones_like(et)
        I_et = torch.where(et>0,zero,et)
        I_et = torch.where(I_et<0,one,I_et)  ## 示性函数 et<0 为1
        logy_hat = alpha_0 + torch.sum(alpha_p*torch.abs(et/torch.pow(sigma2_q,0.5)),dim=1).unsqueeze(dim=1)+\
                             torch.sum(beta_q*torch.log(sigma2_q),dim=1).unsqueeze(dim=1)  + \
                             torch.sum(lever_p*I_et*(et/torch.pow(sigma2_q,0.5)),dim=1).unsqueeze(dim=1)      
        y_hat = torch.exp(logy_hat)
    else:
        raise ValueError
            
        #### 输出估计结果
    all_hat = torch.cat([y_hat, ut_hat_p[:,0].unsqueeze(dim=1)], dim=1)   # all_hat :[batch, 2, out_size]  
        ####   !! 注：y_hat对应t+1时刻;而ut_hat_p对应t时刻
    return all_hat  


###可以进一步考虑huber损失函数  
def reconstruct_target(garch_mode, loss_weights, b_y, b_ys, all_hat, epsilon_std):  
    if garch_mode!="None" and loss_weights:   # 考虑损失函数重新构造的情况 
        if b_y.shape[0]>1:
            b_y_std =  ( ( torch.std(b_ys[:,:,:],axis=(0,1)).reshape(1,2,1) ).\
                        repeat(b_ys.shape[0],1,1)+ epsilon_std ) 
        else:                       # epsilon_std 为稳定因子（考虑了真值、一阶、二阶权重）
            b_y_std = torch.ones_like(b_y)  #防止batch 只有一个样本导致std 缺失
        reconstruct_all_hat= all_hat / b_y_std
        reconstruct_b_y = b_y / b_y_std
    else:        # 仅做相对损失
        reconstruct_all_hat= all_hat / (b_y + epsilon_std)
        reconstruct_b_y = b_y / (b_y + epsilon_std)    
    return reconstruct_all_hat, reconstruct_b_y
    
                      
class Garch_Trans(nn.Module):  # 这一部分并没有新增参数，主要是加一些超参数变量进行二次建模（线性模型）。
    def __init__(self, out_size, d_model=512, nhead=8, dim_feedforward=2048, num_layers=6, dropout=0.1, seq_len=15, \
                 pool="RG", embedding_mode='separately', ar_p = 1, ma_q = 1, garch_mode="None",\
                 loss_weights = False, epsilon_std = 1e-6, train_loss_mode = 0): 
        
        super(Garch_Trans, self).__init__()
        
        # out_sie 为预测步长
        # embedding_mode = 'separately' or 'total' or 'numerical' ,'avg_texg','mix_texg' 分别对应分别嵌入，整体嵌入，仅数值部分嵌入....
        # pool = 'RG' or 'mean'  # 分别对应最后一层 回归是基于 RG位置，还是全局特征平均
        # garch_mode  对应二次建模4种模式
        # ar_p 条件均值滞后阶数
        # ma_q 波动率值滞后阶数
        # loss_weights  False:仅做相对损失 ; True:多损失加权
        # epsilon_std 加权时方差稳定因子
        # train_loss_mode = 0: 波动率 ； range(2): 波动率+条件收益率
        # num_layers 为transformer网络的层数，暂定可选4,6,8,12 
        ###########################################################################################################
           
        #模型的预测步长和窗宽的设定
        self.out_size = out_size
        self.seq_len = seq_len

        # 构造garch模型的超参数
        self.garch_mode = garch_mode
        self.ar_p = ar_p
        self.ma_q = ma_q

        ## 几种损失函数构造
        self.loss_weights = loss_weights
        self.epsilon_std = epsilon_std  # 加权时方差稳定因子
        self.train_loss_mode = train_loss_mode
        
        # 根据num_layers 确定  d_model, nhead, dim_feedforward 
        d_model, nhead, dim_feedforward = trans_layers_paramters(num_layers)   
            
        self.model = TransAm(out_size=out_size, d_model=d_model ,nhead=nhead, dim_feedforward=dim_feedforward, \
                              num_layers=num_layers, dropout=dropout,pool=pool, embedding_mode=embedding_mode, \
                              ar_p = ar_p, ma_q = ma_q, garch_mode=garch_mode)        ## √
        self.loss_func = nn.MSELoss()  # criterion 
    
    
    def predict(self, b_x):
        # 不给是 b_y_previous0，相应位置给None
        # 这里b_y_previous用 b_ys[:,:-1,:,:] 或 b_ys代替表示之前的信息,(不会用到当前信息b_ys[:,-1,:,:]) 
        # 如没有b_y_previous,则可以根据训练集data_max，data_min 还原之前的价格，再导出b_y_previous                 
        
        model = self.model
        out_size = self.out_size
        garch_mode = self.garch_mode
        ar_p = self.ar_p
        ma_q = self.ma_q
      
        output = model(b_x, None)  # input :[seq_len,batch,feature_size] 

        all_hat = all_hat_predict(output=output, out_size=out_size, garch_mode=garch_mode, b_x=b_x, ar_p=ar_p, ma_q=ma_q) ## √
        # all_hat :[batch, 1 ,out_size]   or   [batch, 1+ar_p, out_size]  
        return all_hat


    def forward(self,b_x,b_ys):
        # b_ys : seq_len个时间点序列y的值（真值、一二阶差分等）  [batch,seq,out_size]  or [batch, seq, 1+ difference_order, out_size]
        # b_y : 当前时间点序列y的值（真值、一二阶差分等）
        garch_mode = self.garch_mode
        loss_weights = self.loss_weights
        epsilon_std = self.epsilon_std
        train_loss_mode = self.train_loss_mode
        
        all_hat = self.predict(b_x)  # all_hat：[batch, 1 ,out_size]   or   [batch, 1+ar_p, out_size]  
                         
        b_ret = b_x.permute(1, 0, 2)[:,:,-2].unsqueeze(dim=2)  ## 提取t时刻条件收益率真值    !!!!! outsize只能是1 
        b_ys = torch.cat([b_ys, b_ret], dim=2) ## t+1时刻波动率真值 + t时刻条件收益率真值  [batch, seq, 2] 
        b_y = b_ys[:,-1].unsqueeze(dim=2) # b_y: [batch, 2, out_size]   ##  !!out_size =1
 
        ############# 损失函数的构造#################  
        # 相对损失或 加权损失， loss_weights=false 为 相对损失， True是为加权损
        #reconstruct_all_hat, reconstruct_b_y = reconstruct_target(garch_mode=garch_mode, loss_weights=loss_weights, b_y=b_y,\
                                               #b_ys=b_ys, all_hat=all_hat, epsilon_std=epsilon_std)        
        # 多目标的损失函数的构造  train_loss_mode = 0: 波动率 ； train_loss_mode: range(2): 波动率+条件收益率
        #loss = self.loss_func(reconstruct_all_hat[1:,train_loss_mode], reconstruct_b_y[1:,train_loss_mode])  ## !!!![1:,train_loss_mode]
        
        loss = self.loss_func(all_hat[1:,train_loss_mode], b_y[1:,train_loss_mode]) 
        
        return loss


def get_Coeff_func(output, b_xs ,ar_p,ma_q):  ### output出的参数
    ## output TransAm的输出
    ## garch_mode 不同garch方程  此时只有 garch、torch、egarch
    ## batch_size
    ## ar_p=1,ma_q=1
    ## 提取参数
    ut_hat_p = output[:,range(ar_p)]  ## output range(ar_p)列  为ut_hat(t-1...t-p)条件均值
    alpha_0 = output[:,0+ar_p].unsqueeze(dim=1)  ## output第ar_p列 为alpha0 常数项
    alpha_p = output[:,range(1+ar_p,1+2*ar_p)]## output range(1+ar_p,1+2*ar_p)列 为et_1=(yt_1 - ut_1)^2 的滞后项系数
    beta_q = output[:,range(1+2*ar_p,1+2*ar_p+ma_q)] ## output range(1+2*ar_p,1+2*ar_p+ma_q)列 为sigma2t_1 的滞后项系数
    lever_p = output[:,range(-ar_p,0)] ## output range(-ar_p,0)列 为杠杆效应 的滞后项系数  ！！ 注:只用于tgarch，egarch； garch不用输出该变量
    ## 提取滞后真值
    b_xs = b_xs.unsqueeze(dim=2) 
    ret_p = b_xs[:,range(ar_p)] ##  b_xs range(ar_p)列 为yt_1的滞后收益率数据
    sigma2_q = b_xs[:,range(ar_p,ar_p+ma_q)]##  b_xs range(ar_p,ar_p+ma_q)列 为sigma2t_1的滞后波动率数据
    
    return ut_hat_p, alpha_0, alpha_p, beta_q ,ret_p, sigma2_q, lever_p
    
    
    
    