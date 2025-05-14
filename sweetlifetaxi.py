## Project Description

Sweet Lift Taxi company has collected historical data on taxi orders at airports. To attract more drivers during peak hours, we need to predict the number of taxi orders for the next hour. Build a model for such a prediction.

The RECM metric on the test set should not be higher than 48.

## Project Instructions

1. Download the data and resample for one hour.
2. Analyze the data
3. Train different models with different hyperparameters. The test sample should be 10% of the initial dataset.4. Test the data using the test sample and provide a conclusion.

## Data Description

The data is stored in the file `taxi.csv`.

The number of orders is in the column `num_orders`.
<div class="alert alert-block alert-success">
<b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>

## Preparación
Librerías

import pandas as pd
import math
import numpy as np
import random
import seaborn as sns
from math import sqrt
import matplotlib.pyplot as plt

import sklearn.linear_model
import sklearn.metrics
import sklearn.neighbors
import lightgbm as lgb

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
from statsmodels.tsa.seasonal import seasonal_decompose

from IPython.display import display

import itertools
    
df_taxi=pd.read_csv('/datasets/taxi.csv')
Data Info
df_taxi.info()
df_taxi=pd.read_csv('/datasets/taxi.csv',parse_dates=[0],index_col=[0])
df_taxi.info()
Data Head
df_taxi.head(10)
Null Data
df_taxi.isna().sum()
Duplicated Data
df_taxi.duplicated().sum()
## Análisis
df_taxi.plot()
plt.show()
df_taxi=df_taxi.resample('1H').sum()
df_taxi.plot()
plt.show()
df_descompose=seasonal_decompose(df_taxi)

plt.figure(figsize=(6,8))
plt.subplot(311)
df_descompose.trend.plot(ax=plt.gca())
plt.title('Trend')

plt.figure(figsize=(6,8))
plt.subplot(311)
df_descompose.seasonal.plot(ax=plt.gca())
plt.title('Seasonal')

plt.figure(figsize=(6,8))
plt.subplot(311)
df_descompose.resid.plot(ax=plt.gca())
plt.title('Residual')

plt.tight_layout()
plt.show()

df_descompose.seasonal.loc[(df_descompose.seasonal.index > '1 March 2018') &(df_descompose.seasonal.index < '5 March 2018')].plot()
def characteristics (data,lag_hour,mean_movil_hour):
    data['month']=data.index.month
    data['day']=data.index.day
    data['dayofweek']=data.index.week
    data['hour']=data.index.hour    
    
    for hour in range(1,lag_hour+1):
        data[f'lag_{hour}']=data['num_orders'].shift(hour)
    
    data['rolling_mean']=data['num_orders'].shift().rolling(mean_movil_hour).mean()
characteristics(df_taxi,10,4)
df_taxi.head(10)
df_taxi.dropna(inplace=True)
df_taxi.head()

## Formación
train_valid, test=train_test_split(df_taxi,shuffle=False,test_size=0.2)
train,valid=train_test_split(train_valid,shuffle=False,test_size=0.2)

print(train.index.min(),train.index.max())
print(valid.index.min(),valid.index.max())
print(test.index.min(),test.index.max())
## Prueba
def rmse(true,pred):
    return math.sqrt(mean_squared_error(true,pred))
features_train=train.drop(['num_orders'],axis=1)
target_train=train['num_orders']

features_valid=valid.drop(['num_orders'],axis=1)
target_valid=valid['num_orders']

features_test=test.drop(['num_orders'],axis=1)
target_test=test['num_orders']
<div class="alert alert-block alert-success">
<b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>

Muy bien, dividiste los datos en los sets necesarios y extraíste características extra para darle más información al modelo, esto es un paso importante para asegurar un correcto desempeño
</div>
Modelo Lineal
rl_model=LinearRegression()
rl_model.fit(features_train,target_train)
pred_train=rl_model.predict(features_train)
pred_valid=rl_model.predict(features_valid)
print(f'RSME Train:{rmse(target_train,pred_train)}')
print(f'RSME Valid:{rmse(target_valid,pred_valid)}')

Random Forest Regressor Model
best_model=[]
best_pred=1000000000000000
for tree_numb in range(1,30):
    for depth in range(1,30):
        rf_model=RandomForestRegressor(n_estimators=tree_numb,max_depth=depth)
        rf_model.fit(features_train,target_train)        
        
        pred_train=rf_model.predict(features_train)
        pred_valid=rf_model.predict(features_valid)
        
        #print(f'---- Random Forest with {tree_numb} tree and {depth} of depth ----')
        #print(f'RSME Train:{rmse(target_train,pred_train)}')
        #print(f'RSME Valid:{rmse(target_valid,pred_valid)}')
        #print()
        
        if rmse(target_valid,pred_valid) < best_pred:
            best_model=[tree_numb,depth]
            best_pred=rmse(target_valid,pred_valid)
            
print(f'The best Random Forest is {best_model[0]} trees with {best_model[1]} depth')
Light GBM
lgbm_model=LGBMRegressor(learning_rate=0.1,num_iterations=2000,objetive='rmse')
lgbm_model.fit(features_train,target_train,eval_set=(features_valid,target_valid))

pred_train=lgbm_model.predict(features_train)
pred_valid=lgbm_model.predict(features_valid)

print()
print(f'RSME Train:{rmse(target_train,pred_train)}')
print(f'RSME Valid:{rmse(target_valid,pred_valid)}')
print()
Cat Boost
catboost_model = CatBoostRegressor(learning_rate=0.1,iterations=2000,loss_function='RMSE',verbose=100)
catboost_model.fit(features_train, target_train,eval_set=(features_valid, target_valid), use_best_model=True, early_stopping_rounds=100)

pred_train = catboost_model.predict(features_train)
pred_valid = catboost_model.predict(features_valid)

print()
print(f'RSME Train:{rmse(target_train,pred_train)}')
print(f'RSME Valid:{rmse(target_valid,pred_valid)}')
print()

Testing
pred_test_rl=rl_model.predict(features_test)
pred_test_rf=rf_model.predict(features_test)
pred_test_lgbm=lgbm_model.predict(features_test)
pred_test_catboost=catboost_model.predict(features_test)
print()
print(f'RMSE test Lienal Regresion: {rmse(target_test,pred_test_rl)}')
print(f'RMSE test Random Forest: {rmse(target_test,pred_test_rf)}')
print(f'RMSE test LGBM: {rmse(target_test,pred_test_lgbm)}')
print(f'RMSE test Cat Boost: {rmse(target_test,pred_test_catboost)}')
Conclusion
The CatBoost model is the most suitable for predicting taxi orders based on historical data, providing Sweet Lift Taxi Company with an effective tool to plan and optimize its operations during peak hours. This approach will significantly contribute to improving efficiency and customer experience.
