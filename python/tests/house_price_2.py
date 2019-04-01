import numpy as np
import pandas as pd
from pandas_xbart import PandasXbart
from collections import OrderedDict
import time

#from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def rmse(y1,y2):
	return np.sqrt(np.mean((y1-y2)**2))


full_imputed = pd.read_csv('tests/fullimputed.csv')
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

train_data = pd.concat([cont_train.loc[:1299,],cat_train.loc[:1299,]],axis=1)
valid_data = pd.concat([cont_train.loc[1300:,],cat_train.loc[1300:,]],axis=1)

target_train = np.log1p(target[:1300]); target_valid = np.log1p(target.loc[1300:]);

m = 125
tau = .67*np.var(target_train)/m
params = OrderedDict([('M',m),('L',1),("N_sweeps",50)
							,("Nmin",1),("Ncutpoints",30)
							,("alpha",0.95),("beta",2 ),("tau",tau),("burnin",15),("mtry",7),("max_depth_num",25),
							("draw_sigma",False),("kap",16),("s",4),("verbose",False),("m_update_sigma",True),
							("draw_mu",False),("parallel",True)])

xbart = PandasXbart(params)
start_1 = time.time()
print("Cat Shape " + str(cat_train.shape[1]))
xbart.fit(train_data_norm,target_train,cat_train.shape[1])

# print("Xbart rmse:" + str(rmse(y_hat,target_valid)))
# print("Xbart fit time:" + str(XBART_time_fit))
# print("Xbart predict time:" + str(XBART_time_predict))




