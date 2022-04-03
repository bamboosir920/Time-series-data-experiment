import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#从pyplot导入MultipleLocator类,这个类用于设置刻度间隔
from matplotlib.pyplot import MultipleLocator
from math import *
#避免中文显示不出来
matplotlib.rc("font",family='KaiTi')
#避免负号显示不出来
matplotlib.rcParams['axes.unicode_minus']=False
#tsa为Time Series analysis缩写
import statsmodels.api as sm
import scipy.stats as scs
import seaborn as sns
# 导入 arch 包中的 arch_model 模块
from arch import arch_model

'''
用来绘制基础随时间而变化趋势图

date:时间,x轴
data:数据,y轴,具体数值
title:标题
xlabel:横坐标注释
ylabel:纵坐标注释
x_loc:X轴默认刻度,默认值为10
y_loc:y轴默认刻度,默认值为10
colors:折线颜色,默认为'#0066cc'
'''
def baseimg(data,date,title='',xlabel='',ylabel='',x_loc=10,y_loc=10,colors='#0066cc'):
    #绘制网格
    plt.grid(alpha=0.6,linestyle=':')
    #作图
    plt.plot(date,data,color=colors)
    #把x轴的刻度间隔设置为500,并存在变量里
    x_major_locator=MultipleLocator(x_loc)
    #把y轴的刻度间隔设置为500,并存在变量里
    y_major_locator=MultipleLocator(y_loc)
    #ax为两条坐标轴的实例
    ax=plt.gca()
    #把x轴的主刻度设置为500的倍数
    ax.xaxis.set_major_locator(x_major_locator)
    #把y轴的主刻度设置为500的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    #设置标题、x轴注释、y轴注释
    plt.title(title,fontsize=16)
    plt.xlabel(ylabel,fontsize=12)
    plt.ylabel(xlabel)
    plt.show()
    return 0

'''
Logarithmic return rate
对数收益率
对数收益率: 所有价格取对数后两两之间的差值。

input:
data:数据
is_img:是否作图,True/False
title:标题名
colors:折线颜色,默认为'#0066cc'
'''
def log_return_rate(data,is_img=True,title='Logarithmic return rate',colors='#0066cc'):
    logReturn = np.diff(np.log(data))
    if is_img==True:
        plt.plot(logReturn,color=colors)
        plt.title(title,fontsize=16)
        plt.show()
    return logReturn

'''
simple return of rate
简单收益率
input:
data:数据
is_img:是否作图,True/False
title:标题名
colors:折线颜色,默认为'#0066cc'
'''
def simple_return_rate(data,is_img=True,title='simple return of rate',colors='#0066cc'):
    simpleReturn = np.diff(data)
    if is_img==True:
        plt.plot(simpleReturn,color=colors)
        plt.title(title,fontsize=16)
        plt.show()
    return simpleReturn

'''
价格预测
input:
data:价格序列
ret:收益率
count:拟合次数
title:标题
'''
def predict(data,ret,count,title=''):
    # 设置一个空列表来保存我们每个模拟价格序列的最终值
    result = []

    S = data[-1] #起始股票价格（即最后可用的实际股票价格）
    T = 252 #交易天数
    all=0

    mu=np.mean(ret) #收益率
    vol=np.std(ret)*sqrt(252/len(data)) #波动率


    #选择要模拟的运行次数-我选择了10,000
    for i in range(count):
    #使用随机正态分布创建每日收益表
        daily_returns=np.random.normal((1+mu)**(1/T),vol/sqrt(T),T)
        #设定起始价格，并创建由上述随机每日收益生成的价格序列
    price_list = [S]
    for x in daily_returns:
        price_list.append(price_list[-1]*x)
        #将每次模拟运行的结束值添加到我们在开始时创建的空列表中
        result.append(price_list[-1])

    plt.figure(figsize=(10,6))
    plt.hist(result,bins= 100)
    plt.axvline(np.percentile(result,5), color='r', linestyle='dashed', linewidth=2)
    plt.axvline(np.percentile(result,95), color='r', linestyle='dashed', linewidth=2)
    mean=np.mean(result)
    a5=result[int(len(result)/5)]
    a95=result[-int(len(result)/5)]
    print(mean)
    print(a95)
    print(a5)
    plt.figtext(0.8,0.8,s="起始价格: %.2f元" %S)
    plt.figtext(0.8,0.7,"平均价格 %.2f 元" %mean)
    plt.figtext(0.8,0.6,"5%"+" 置信度: %.2f元" %a5)
    plt.figtext(0.15,0.6, "95%" +"置信度: %.2f元" %a95)
    plt.title(title, weight='bold', fontsize=12)
    plt.show()
    

'''
频率分布图
Histogram of frequency distribution
input:
data:数据
title:标题
'''
def frequency_distribution(data,title='频率分布图'):
    sns.distplot(data, color='blue') #密度图
    plt.title(title,fontsize=16)
    plt.show()
    
'''
grach建模
input:
data:数据
title1:拟合ARCH残差图标题
title2:条件方差图标题
vol (str, optional) 波动率模型的名称，目前支持: 'GARCH' （默认）, 'ARCH', 'EGARCH', 'FIARCH' 以及 'HARCH'。
p (int, optional) 对称随机数的滞后阶（译者注：即扣除均值后的部分）。
o (int, optional)  非对称数据的滞后阶
q (int, optional)  波动率或对应变量的滞后阶
power (float, optional)  使用GARCH或相关模型的精度
dist (int, optional) 误差分布的名称，目前支持下列分布：
    正态分布: 'normal', 'gaussian' (default)
    学生T分布: 't', 'studentst'
    偏态学生T分布: 'skewstudent', 'skewt'
    通用误差分布: 'ged', 'generalized error”
'''
def Arch(data,p_=2,q_=2,o_=1,power_=2.0,vol_='Garch',dist_='StudentsT',title1='对比图',title2='拟合残差',title3='条件方差'):
     # 设定模型
    am=arch_model(data, p=p_, q=q_, o=o_,power=power_, vol=vol_, dist=dist_)
    res = am.fit(update_freq=5, disp='off')
    print(res.summary())

    fig = res.hedgehog_plot(type='mean')
    plt.title(title1,size=15)
    plt.show()

    plt.plot(res.resid)
    plt.title(title2,size=15)
    plt.show()
    
    plt.plot(res.conditional_volatility,color='r')
    plt.title(title3,size=15)
    plt.show()
