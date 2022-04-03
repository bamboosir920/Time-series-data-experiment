#格式 日期 价格(元/吨)
import numpy as np
import csv
from TStest import *
from TSbase import *


sns.set()  #切换到sns的默认运行配置
#避免中文显示不出来
matplotlib.rc("font",family='KaiTi')
#避免负号显示不出来
matplotlib.rcParams['axes.unicode_minus']=False
#导入数据
p=r'./xxx.csv'
with open(p) as f:
    data=np.loadtxt(f,str,delimiter=',') 
x=np.array(data[:,0])
y=np.array(data[:,1])
y = y.astype(np.float64)



'''
基础作图
'''
baseimg(data=y,date=x,xlabel="时间",ylabel="收盘价",x_loc=500,y_loc=500)

'''
收益率
'''
# 简单收益率
simpleReturn=simple_return_rate(y,False)
# 对数收益率
logReturn=log_return_rate(y,False)

'''
JB检验
'''
print(JBtest(logReturn))

'''
平稳性检验
'''
TS_ADF(logReturn)
TS_ACF(logReturn)
TS_PACF(logReturn)

'''
大图
# '''
ts_plot(logReturn,lags=30,title='对数收益率')

'''
白噪声检验
'''
white_test(simpleReturn,12)
white_test(logReturn,30)

'''
频率分布图
'''
frequency_distribution(logReturn,title='频率分布图')

'''
Garch建模
'''
Arch(logReturn)

'''
预测
'''
predict(y,logReturn,20000,'..')


