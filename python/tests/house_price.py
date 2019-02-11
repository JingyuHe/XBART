import numpy as np
import pandas as pd
from abarth import Abarth
from collections import OrderedDict
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

cat =  dummied.select_dtypes(include=['int'])
cat_train = cat.loc[~idx]
cat_test = cat.loc[idx]



train_data = pd.concat([cont_train.loc[:1299,]+np.random.normal(scale=0.0001,
						size=cont_train.loc[:1299,].shape),cat_train.loc[:1299,]],axis=1)
valid_data = pd.concat([cont_train.loc[1300:,] +np.random.normal(scale=0.0001,
						size=cont_train.loc[1300:,].shape) ,cat_train.loc[1300:,]],axis=1)


# train_data = pd.concat([cont_train.loc[:1299,],cat_train.loc[:1299,]],axis=1)
# valid_data = pd.concat([cont_train.loc[1300:,] ,cat_train.loc[1300:,]],axis=1)

target_train = np.log1p(target[:1300]); target_valid = np.log1p(target.loc[1300:]);


train_data_norm = train_data # pd.concat([cont_train.loc[:1299,],cat_train.loc[:1299,]],axis=1) + np.random.normal(scale=0.0001,size=train_data.shape)
valid_data_norm = valid_data # pd.concat([cont_train.loc[1300:,] ,cat_train.loc[1300:,]],axis=1)+ np.random.normal(scale=0.0001,size=valid_data.shape)

# train_data_norm=pd.concat([cont_train.loc[:1299,],cat_train.loc[:1299,]],axis=1) + np.random.normal(scale=0.0001,size=train_data.shape)
# valid_data_norm=pd.concat([cont_train.loc[1300:,] ,cat_train.loc[1300:,]],axis=1)+ np.random.normal(scale=0.0001,size=valid_data.shape)

train_data.nunique(axis=1)

m = 125
tau = .67*np.var(target_train)/m
params = OrderedDict([('M',m),('L',1),("N_sweeps",50)
							,("Nmin",1),("Ncutpoints",30)
							,("alpha",0.95),("beta",2 ),("tau",tau),("burnin",15),("mtry",7),("max_depth_num",25),
							("draw_sigma",False),("kap",16),("s",4),("verbose",False),("m_update_sigma",True),
							("draw_mu",False),("parallel",False)])

# m = 300
# tau = .67*np.var(target_train)/m
# params = OrderedDict([('M',m),('L',1),("N_sweeps",70)
# 							,("Nmin",1),("Ncutpoints",30)
# 							,("alpha",0.95),("beta",2 ),("tau",tau),("burnin",15),("mtry",7),("max_depth_num",25),
# 							("draw_sigma",False),("kap",16),("s",4),("verbose",False),("m_update_sigma",False),
# 							("draw_mu",False),("parallel",False)])


print("Abarth Fit Predict Seperate...")
xbart_2 = Abarth(params)
start_1 = time.time()
xbart_2.fit(train_data_norm.values,target_train.values,cat_train.shape[1])
end_1 = time.time()
start_2 = time.time()
y_pred_2 = xbart_2.predict(valid_data_norm.values)
end_2 = time.time()
abarth_time_fit = end_1-start_1
abarth_time_predict = end_2-start_2
y_hat_xbart_2 = y_pred_2[:,params["burnin"]:].mean(axis=1)

print("Abarth...")
xbart = Abarth(params)
start = time.time()
y_pred = xbart.fit_predict(train_data_norm.values,target_train.values,
	valid_data_norm.values,cat_train.shape[1])
end = time.time()
abarth_time = end-start
y_hat_xbart = y_pred[:,params["burnin"]:].mean(axis=1)


print("Boosting...")
bst = GradientBoostingRegressor()
start = time.time()
bst.fit(train_data,target_train)
y_hat_bst = bst.predict(valid_data)
end = time.time()
bst_time= end-start

print("RandomForest...")
rf = RandomForestRegressor(n_estimators=500)
start = time.time()
rf.fit(train_data,target_train)
y_hat_rf= rf.predict(valid_data)
end = time.time()
rf_time = end-start


print("Xbart rmse:" + str(rmse(y_hat_xbart,target_valid)))
print("Xbart 2 rmse:" + str(rmse(y_hat_xbart_2,target_valid)))
print("Boosting rmse:" + str(rmse(y_hat_bst,target_valid)))
print("RandomForest rmse:" + str(rmse(y_hat_rf,target_valid)))

print("Xbart time:" + str(abarth_time))
print("Xbart time fit:" + str(abarth_time_fit))
print("Xbart time predict:" + str(abarth_time_predict))
print("Xbart time seperate:" + str(abarth_time_fit+abarth_time_predict))
print("Boosting time:" + str(bst_time))
print("RandomForest time:" + str(rf_time))

# train_data_norm.to_csv("house_train.csv");
# valid_data_norm.to_csv("house_valid.csv");
# target_train.to_csv("target_train.csv"); target_valid .to_csv("target_valid.csv");


