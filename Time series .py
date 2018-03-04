# -*- coding: utf-8 -*-
"""
Created on Sat Mar 03 21:45:41 2018

@author: LIU
"""

#smooth

import pandas as pd
import numpy as np
#import data
datafile='d:/data/discdata.xls'

data=pd.read_excel(datafile,encoding='utf-8')
data=data[data['TARGET_ID']==184].copy()

#preprocess the data
group_data=data.groupby('COLLECTTIME')

def reform(x):
    result=pd.Series(index=['sys_name','cwxt_bd:184:c:\\','cwxt_bd:184:d:\\','collecttime'])
    result['sys_name']=x['SYS_NAME'].iloc[0]
    result['cwxt_bd:184:c:\\']=x['VALUE'].iloc[0]
    result['cwxt_bd:184:d:\\']=x['VALUE'].iloc[1]
    result['collecttime']=x['COLLECTTIME'].iloc[0]
    return result

data_processed=group_data.apply(reform)
print(data_processed.head())

#plot the timeseries plot
column=['cwxt_bd:184:d:\\']    
data_d=data_processed[column]
column2=['cwxt_bd:184:c:\\']    
data_c=data_processed[column2]

import matplotlib.pyplot as plt
plt.plot(data_d)
plt.plot(data_c)


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput
    
test_stationarity(data_d) 
 
# check the stationality(find the best diff value)
data_time=data_processed.iloc[:len(data_processed)-5]
from statsmodels.tsa.stattools import adfuller as ADF
diff=0
adf=ADF(data_time['cwxt_bd:184:d:\\'])
while adf[1]>=0.05:
    diff=diff+1
    adf=ADF(data_time['cwxt_bd:184:d:\\'].diff(diff).dropna())
print('the original is smooth after %s diff, p_value is %s'%(diff,adf[1]))

plt.plot(data_time['cwxt_bd:184:d:\\'].diff(1))

diff=0
adf=ADF(data_time['cwxt_bd:184:c:\\'])
while adf[1]>=0.05:
    diff=diff+1
    adf=ADF(data_time['cwxt_bd:184:c:\\'].diff(diff).dropna())
print('the original is smooth after %s diff, p_value is %s'%(diff,adf[1]))

plt.plot(data_time['cwxt_bd:184:c:\\'].diff(1))

data_station=pd.DataFrame()
data_station=data_time

#check the while noise 
from statsmodels.stats.diagnostic import acorr_ljungbox
[[lb], [p]] = acorr_ljungbox(data_time['cwxt_bd:184:d:\\'], lags = 1)
if p < 0.05:
  print('original data is not white noise series，p-value is %s' %p)
else:
  print('original data is white noise series，p-value is %s' %p)
a=data_time['cwxt_bd:184:d:\\'].diff(1).dropna()
[[lb1], [p1]] = acorr_ljungbox(a.dropna(),lags=1)
if p1 < 0.05:
  print('one diff data is not white noise data：%s' %p)
else:
  print('one diff data is not white noise data：%s' %p)
  
#fit the model and find the p,d,q
data_fit = data_processed.iloc[: len(data)-5] 
xdata = data_fit['cwxt_bd:184:d:\\']
from statsmodels.tsa.arima_model import ARIMA
pmax = int(len(xdata)/10)# normal no more than 10
qmax = int(len(xdata)/10)# normal no more than 10
bic_matrix = [] #bic matrix
for p in range(pmax+1):
  tmp = []
  for q in range(qmax+1):
    try: # there may exist error so use try to skip the error
      tmp.append(ARIMA(xdata, (p,1,q)).fit().bic)
    except:
      tmp.append(None)
  bic_matrix.append(tmp)

bic_matrix = pd.DataFrame(bic_matrix) #can find the min from matrix

p,q = bic_matrix.stack().idxmin() # use stack the seperate and them use idxmin find the min 
print('BIC MIN P AND Q VALUE IS：%s、%s' %(p,q))


#check the model
lagnum = 12 #the number of lag

from statsmodels.tsa.arima_model import ARIMA #ARIMA(0,1,1)

arima = ARIMA(xdata, (0, 1, 1)).fit() #bulid and train
xdata_pred = arima.predict(typ = 'levels') #predict
pred_error = (xdata_pred - xdata).dropna() #error

from statsmodels.stats.diagnostic import acorr_ljungbox #white noise

lb, p= acorr_ljungbox(pred_error, lags = lagnum)
h = (p < 0.05).sum() #when p-value smaller than, it is no while noise
if h > 0:
  print('ARIMA(0,1,1) is not white noise')
else:
  print('ARIMA(0,1,1) is white noise')



abs_ = (xdata_pred - xdata).abs()
mae_ = abs_.mean() # mae
rmse_ = ((abs_**2).mean())**0.5 # rmse
mape_ = (abs_/xdata).mean() # mape

print('mape: %0.4f，\n mse：%0.4f，\n rmse：%0.6f。' %(mae_, rmse_, mape_))


