import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score,log_loss
import xgboost as xgb
from xbart import XBART
from scipy.stats import norm

print("Read Data")
train_fe = pd.read_csv("train_fe.csv")
valid_fe = pd.read_csv("valid_fe.csv")

print("Fit XBART Hack")
model = XBART(num_trees = 15,num_sweeps = 150,n_min = 30,burnin = 25)
model.fit(train_fe.drop("Survived",axis=1),train_fe["Survived"]*2-1)
preds_hack = model.predict(valid_fe.drop("Survived",axis=1),return_mean=False)
phat_test_hack = preds_hack.mean(axis=1)
xbart_hack_score = accuracy_score(valid_fe['Survived'],phat_test_hack>0)
xbart_hack_log = log_loss(valid_fe['Survived'],norm.cdf(phat_test_hack))

print("Fit RF")
rf = RandomForestClassifier(n_estimators=20) 
rf.fit(train_fe.drop("Survived",axis=1),train_fe["Survived"])
rf_score = rf.score(valid_fe.drop("Survived",axis=1),valid_fe["Survived"])
rf_pred = rf.predict_proba(valid_fe.drop("Survived",axis=1))[:,1]
rf_log = log_loss(valid_fe['Survived'],rf_pred)


print("Fit XGB")
bst = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators = 30,colsample_bytree=.8,importance="gain")
bst.fit(train_fe.drop("Survived",axis=1),train_fe["Survived"])
xgb_score = bst.score(valid_fe.drop("Survived",axis=1),valid_fe["Survived"])
xgb_pred = bst.predict_proba(valid_fe.drop("Survived",axis=1))[:,1]
xgb_log = log_loss(valid_fe['Survived'],xgb_pred)


print("XBART Hack: acc: " + str(xbart_hack_score) + " log_loss: " + str(xbart_hack_log) )
print("RF acc: " + str(rf_score) + " log_loss: " + str(rf_log))
print("XGB acc"+ str(xgb_score)  + "log_loss: " + str(xgb_log))