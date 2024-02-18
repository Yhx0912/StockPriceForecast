import matplotlib.pyplot as plt
import numpy as np
import os


## 收益率趋势图
def Yield_Rate_Trend(X,X_test,stockname,savefig=False,fig_dir=None,fig_name=None):
    x_all = np.arange(len(X)+len(X_test))
    x1 = np.arange(len(X))
    x2 = np.arange(len(X),len(X)+len(X_test))
    plt.plot(x1,X[:,0],label='Train',color='blue')
    plt.plot(x2,X_test[:,0],label='Test',color='green')
    plt.ylabel('Yield')
    plt.ylim((min(min(X[:,0]),min(X_test[:,0]))-0.1),max(max(X[:,0]),max(X_test[:,0]))+0.25)
    title = stockname + " - Yield Rate Trend Chart"
    plt.title(title)
    plt.legend()
    if savefig:
        if fig_dir==None: fig_dir = os.getcwd()
        plt.savefig(os.path.join(fig_dir,fig_name))
    plt.show()
    

## 波动率趋势图
def Volatility_Rate_Trend(y_multi,y_multi_test,stockname,savefig=False,fig_dir=None,fig_name=None):
    x_all = np.arange(len(y_multi)+len(y_multi_test))
    x1 = np.arange(len(y_multi))
    x2 = np.arange(len(y_multi),len(y_multi)+len(y_multi_test))
    plt.plot(x1,y_multi[:,0],label='Train',color='blue')
    plt.plot(x2,y_multi_test[:,0],label='Test',color='green')
    plt.ylabel('Volatility')
    title = stockname + " - Volatility Forecast Chart"
    plt.title(title)
    plt.legend()
    if savefig:
        if fig_dir==None: fig_dir = os.getcwd()
        plt.savefig(os.path.join(fig_dir,fig_name))
    plt.show()

## 波动率预测图
def Volatility_Forecast(y_true,y_predict,stockname,savefig=False,fig_dir=None,fig_name=None):
    x = np.arange(len(y_true))
    plt.plot(x, y_true, label='True', color = "red",  linestyle='-')
    plt.plot(x, y_predict, label='Predict',color = "blue",  linestyle='--')
    #plt.xlabel('iterations (x' + 'epoch' + ')')
    plt.ylabel('Volatility')
    title = stockname + " - Volatility Forecast Chart"
    plt.title(title)
    plt.legend()
    if savefig:
        if fig_dir==None: fig_dir = os.getcwd()
        plt.savefig(os.path.join(fig_dir,fig_name))
    plt.show()
