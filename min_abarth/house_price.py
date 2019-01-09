import numpy as np
import pandas as pd
from abarth import Abarth
from collections import OrderedDict
#from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def rmse(y1,y2):
	return np.sqrt(np.mean((y1-y2)**2))

path = "/Users/saaryalov/DataScienceClubASU/kaggle-house-prices/"
full_imputed = pd.read_csv(path+'fullimputed.csv')
del full_imputed['Unnamed: 0']
dummied = pd.get_dummies(full_imputed)

idx = dummied["SalePrice"].isnull()
cont = dummied.select_dtypes(include=['float'])
cont_train = cont.loc[~idx]
cont_test= cont.loc[idx]
target= cont_train["SalePrice"];del cont_train["SalePrice"]
del cont_test["SalePrice"]

cat =  dummied.select_dtypes(include=['int'])
cat_train = cat.loc[~idx]
cat_test = cat.loc[idx]



train_data = pd.concat([cont_train.loc[:1299,]+np.random.normal(scale=0.0001,
						size=cont_train.loc[:1299,].shape),cat_train.loc[:1299,]],axis=1)
valid_data = pd.concat([cont_train.loc[1300:,] +np.random.normal(scale=0.0001,
						size=cont_train.loc[1300:,].shape) ,cat_train.loc[1300:,]],axis=1)
target_train = np.log1p(target[:1300]); target_valid = np.log1p(target.loc[1300:]);


train_data_norm = train_data #+ np.random.normal(scale=0.0001,size=train_data.shape)
valid_data_norm = valid_data #+ np.random.normal(scale=0.0001,size=valid_data.shape)

train_data.nunique(axis=1)

m = 500
tau = .67*np.var(target_train)/m
params = OrderedDict([('M',m),('L',1),("N_sweeps",40)
							,("Nmin",1),("Ncutpoints",30)
							,("alpha",0.95),("beta",5 ),("tau",tau),("burnin",15),("mtry",7),("max_depth_num",50),
							("draw_sigma",False),("kap",16),("s",4),("verbose",False),("m_update_sigma",False),
							("draw_mu",False),("parallel",False)])

xbart = Abarth(params)
y_pred = xbart.fit_predict_2d_all(train_data_norm.values,target_train.values,valid_data_norm.values,cat_train.shape[1])
y_hat_xbart = y_pred[:,params["burnin"]:].mean(axis=1)

bst = GradientBoostingRegressor()
bst.fit(train_data,target_train)
y_hat_bst = bst.predict(valid_data)

rf = RandomForestRegressor()
rf.fit(train_data,target_train)
y_hat_rf= rf.predict(valid_data)


print("Xbart rmse:" + str(rmse(y_hat_xbart,target_valid)))
print("Boosting rmse:" + str(rmse(y_hat_bst,target_valid)))
print("RandomForest rmse:" + str(rmse(y_hat_rf,target_valid)))



