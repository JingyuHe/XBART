import numpy as np
import pandas as pd
from xbart import XBART
import time

#from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def rmse(y1,y2):
	return np.sqrt(np.mean((y1-y2)**2))


full_imputed = pd.read_csv('fullimputed.csv')
del full_imputed['Unnamed: 0']
dummied = pd.get_dummies(full_imputed)

idx = dummied["SalePrice"].isnull()
cont = dummied.select_dtypes(include=['float'])
cont_train = cont.loc[~idx]
cont_test= cont.loc[idx]
target= cont_train["SalePrice"];del cont_train["SalePrice"]
del cont_test["SalePrice"]

cat =  dummied.select_dtypes(include=['integer'])
cat_train = cat.loc[~idx]
cat_test = cat.loc[idx]

train_data = pd.concat([cont_train.loc[:1299,],cat_train.loc[:1299,]],axis=1)
valid_data = pd.concat([cont_train.loc[1300:,],cat_train.loc[1300:,]],axis=1)

target_train = np.log1p(target[:1300]); target_valid = np.log1p(target.loc[1300:]);


xbart = XBART(num_trees = 50,tau =1/50,beta = 1.25,num_sweeps = 60,num_cutpoints = 10)
time_start_fit = time.time()
xbart.fit(train_data,target_train,cat_train.shape[1])
time_start_predict = time.time()
y_pred = xbart.predict(valid_data,return_mean=False)
time_end_predict = time.time()
y_hat_xbart = y_pred[:,15:].mean(axis=1)
print("Done!")
print("Xbart rmse:" + str(rmse(y_hat_xbart,target_valid)))
print("Xbart fit time:" + str(time_start_predict - time_start_fit))
print("Xbart predict time:" + str(time_end_predict - time_start_predict))




