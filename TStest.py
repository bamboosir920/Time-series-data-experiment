import pandas as pd
import statsmodels.tsa.api as smt     
#tsa为Time Series analysis缩写
import statsmodels.api as sm
import scipy.stats as scs
from arch import arch_model
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from statsmodels.stats.diagnostic import acorr_ljungbox

#从pyplot导入MultipleLocator类,这个类用于设置刻度间隔
from matplotlib.pyplot import MultipleLocator
#引入单位根检验
from arch.unitroot import ADF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import numpy as np
import scipy.stats as stats
#避免中文显示不出来
matplotlib.rc("font",family='KaiTi')
#避免负号显示不出来
matplotlib.rcParams['axes.unicode_minus']=False


'''
做一个完整检验的大图

input:
data:输入y轴数值
lags:延迟
title:标题

'''
def ts_plot(data, lags=None,title=''):    
    if not isinstance(data, pd.Series):           
        data = pd.Series(data)    
        #matplotlib官方提供了五种不同的图形风格，    
        # #包括bmh、ggplot、dark_background、    
        # #fivethirtyeight和grayscale    
        with plt.style.context('ggplot'):            
            fig = plt.figure(figsize=(10, 8))        
            layout = (3, 2)        
            ts_ax = plt.subplot2grid(layout, (0,0),colspan=2)        
            acf_ax = plt.subplot2grid(layout, (1, 0))        
            pacf_ax = plt.subplot2grid(layout, (1, 1))        
            qq_ax = plt.subplot2grid(layout, (2, 0))        
            pp_ax = plt.subplot2grid(layout, (2, 1))        
            data.plot(ax=ts_ax)        
            ts_ax.set_title(title+'时序图')        
            smt.graphics.plot_acf(data, lags=lags,ax=acf_ax, alpha=0.5)        
            acf_ax.set_title('自相关系数')        
            smt.graphics.plot_pacf(data, lags=lags,ax=pacf_ax, alpha=0.5)        
            pacf_ax.set_title('偏自相关系数')        
            sm.qqplot(data, line='s', ax=qq_ax)        
            qq_ax.set_title('QQ 图')                
            scs.probplot(data, sparams=(data.mean(),data.std()), plot=pp_ax)        
            pp_ax.set_title('PP 图')         
            plt.tight_layout()
            plt.show()    
            return

'''
时间序列稳定性检验,单位根ADF

input:
data:输入y轴数值
name:数值含义

output:
Test Statistic : T值,表示T统计量
p-value: p值,表示T统计量对应的概率值
Lags Used:表示延迟
Number of Observations Used: 表示测试的次数
Critical Value 1% : 表示t值下小于 - 4.938690 , 则原假设发生的概率小于1%, 其它的数值以此类推。

其中t值和p值是最重要的,其实这两个值是等效的,既可以看t值也可以看p值。
p值越小越好,要求小于给定的显著水平,p值小于0.05,等于0最好。
t值,ADF值要小于t值,1%, 5%, 10% 的三个level,都是一个临界值,如果小于这个临界值,说明拒绝原假设。
1%、%5、%10不同程度拒绝原假设的统计值和ADF Test result的比较,ADF Test result同时小于1%、5%、10%即说明非常好地拒绝该假设.
P-value是否非常接近0.
'''
def TS_ADF(data):
    print("**************************************************************************")
    print("**************************************************************************")
    print("Time Series test————ADF")
    res=np.array(ADF(data))
    print(res)
    print("*************************************************************************")
    print("*************************************************************************")
    return res

'''
时间序列稳定性检验,自相关检验ACF

input:
data:输入数据

结果含义:
ACF 是一个完整的自相关函数,可为我们提供具有滞后值的任何序列的自相关值。
简单来说,它描述了该序列的当前值与其过去的值之间的相关程度。
时间序列可以包含趋势,季节性,周期性和残差等成分。
ACF在寻找相关性时会考虑所有这些成分。
直观上来说,ACF 描述了一个观测值和另一个观测值之间的自相关,包括直接和间接的相关性信息。

截尾:在大于某个常数k后快速趋于0为k阶截尾
拖尾:始终有非零取值,不会在k大于某个常数后就恒等于零(或在0附近随机波动)
'''
def TS_ACF(data):
    plot_acf(data)
    plt.show()

'''
时间序列稳定性检验,偏自相关函数PACF

input:
data:输入数据

结果含义:
PACF 是部分自相关函数或者偏自相关函数。
基本上,它不是找到像ACF这样的滞后与当前的相关性,而是找到残差（在去除了之前的滞后已经解释的影响之后仍然存在）与下一个滞后值的相关性。
因此,如果残差中有任何可以由下一个滞后建模的隐藏信息,我们可能会获得良好的相关性,并且在建模时我们会将下一个滞后作为特征。
请记住,在建模时,我们不想保留太多相互关联的特征,因为这会产生多重共线性问题。因此,我们只需要保留相关功能。

截尾:在大于某个常数k后快速趋于0为k阶截尾
拖尾:始终有非零取值,不会在k大于某个常数后就恒等于零(或在0附近随机波动)
'''
def TS_PACF(data): 
    plot_pacf(data)
    plt.show()

'''
白噪声检验
'''
def white_test(data,lag=25):
    print(acorr_ljungbox(data, lags = lag,boxpierce=True))
    return acorr_ljungbox(data, lags = lag,boxpierce=True)

'''
JB检验：
input:
data:输入数据、序列

output：
偏度、峰值、JB检验
'''

def JBtest(data):
    # 样本规模n
    n = data.size
    data_ = data - data.mean()
    """
    M2:二阶中心钜
    skew 偏度 = 三阶中心矩 与 M2^1.5的比
    krut 峰值 = 四阶中心钜 与 M2^2 的比
    """
    M2 = np.mean(data_**2)
    skew =  np.mean(data_**3)/M2**1.5
    krut = np.mean(data_**4)/M2**2

    """
    计算JB统计量，以及建立假设检验
    """
    JB = n*(skew**2/6 + (krut-3 )**2/24)
    pvalue = 1 - stats.chi2.cdf(JB,df=2)
    print("偏度：",stats.skew(data),skew)
    print("峰值：",stats.kurtosis(data)+3,krut)
    print("JB检验：",stats.jarque_bera(data))
    return np.array([JB,pvalue])