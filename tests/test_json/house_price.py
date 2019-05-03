import numpy as np
import pandas as pd
from xbart import XBART
import time

#from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def rmse(y1,y2):
	return np.sqrt(np.mean((y1-y2)**2))


train_data = pd.read_csv('train.csv')
valid_data = pd.read_csv('test.csv')

target_train = train_data["target"]
target_valid = valid_data["target"]

train_data.drop(["target"],axis=1,inplace=True)
valid_data.drop(["target"],axis=1,inplace=True)

xbart = XBART(num_trees = 125,tau = 1/125,beta = 2.0)
time_start_fit = time.time()
xbart.fit(train_data,target_train,p_cat=289)
time_start_predict = time.time()
y_hat_xbart = xbart.predict(valid_data)
time_end_predict = time.time()

print("Done!")
print("Xbart rmse:" + str(rmse(y_hat_xbart,target_valid)))
print("Xbart fit time:" + str(time_start_predict - time_start_fit))
print("Xbart predict time:" + str(time_end_predict - time_start_predict))

xbart.to_json("model.xbart")

xbart2 = XBART(num_trees = 125,tau = 1/125,beta = 2.0)
xbart2.from_json("model.xbart")
y_hat_xbart_2 = xbart2.predict(valid_data)
print("Xbart rmse loaded:" + str(rmse(y_hat_xbart_2,target_valid)))





