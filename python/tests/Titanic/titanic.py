import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xbart import XBART

print("Read Data")
train_fe = pd.read_csv("train_fe.csv")
valid_fe = pd.read_csv("valid_fe.csv")

print("Fit XBART")
model = XBART(num_trees = 10,num_sweeps = 150,n_min = 10,burnin = 25,model = "CLT")
model.fit(train_fe.drop("Survived",axis=1),train_fe["Survived"]*2-1)
preds = model.predict(valid_fe.drop("Survived",axis=1))
phat_test = preds.mean(axis=1)
xbart_score = accuracy_score(valid_fe['Survived'],phat_test>0)


print("Fit RF")
rf = RandomForestClassifier(n_estimators=20) 
rf.fit(train_fe.drop("Survived",axis=1),train_fe["Survived"])
rf_score = rf.score(valid_fe.drop("Survived",axis=1),valid_fe["Survived"])


print("Fit XGB")
bst = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators = 30,colsample_bytree=.8,importance="gain")
bst.fit(train_fe.drop("Survived",axis=1),train_fe["Survived"])
xgb_score = bst.score(valid_fe.drop("Survived",axis=1),valid_fe["Survived"])

print(f"XBART: {xbart_score}" )
print(f"RF: {rf_score}" )
print(f"XGB: {xgb_score}" )