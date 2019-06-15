import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xbart import XBART
from scipy.stats import norm

print("Read Data")
train_fe = pd.read_csv("train_fe.csv")
valid_fe = pd.read_csv("valid_fe.csv")

print("Fit XBART Hack")
model = XBART(num_trees = 15,num_sweeps = 150,n_min = 5,burnin = 25)
model.fit(train_fe.drop("Survived",axis=1),train_fe["Survived"]*2-1)
preds_hack = model.predict(valid_fe.drop("Survived",axis=1),return_mean=False)
phat_test_hack = preds_hack.mean(axis=1)
xbart_hack_score = accuracy_score(valid_fe['Survived'],phat_test_hack>0)

print("Fit XBART CLT")
model = XBART(num_trees = 15,num_sweeps = 150,n_min = 5,burnin = 25,model = "CLT")
model.fit(train_fe.drop("Survived",axis=1),train_fe["Survived"]*2-1)
preds_clt = model.predict(valid_fe.drop("Survived",axis=1),return_mean=False)
phat_test_clt = preds_clt.mean(axis=1)
xbart_clt_score = accuracy_score(valid_fe['Survived'],phat_test_clt>0)

print("Fit XBART Probit")
model = XBART(num_trees = 15,num_sweeps = 150,n_min = 5,burnin = 25,model = "Probit")
model.fit(train_fe.drop("Survived",axis=1),train_fe["Survived"]*2-1)
preds_probit = model.predict(valid_fe.drop("Survived",axis=1),return_mean=False)
phat_test_probit = preds_probit.mean(axis=1)
xbart_probit_score = accuracy_score(valid_fe['Survived'],norm.cdf(phat_test_probit)>0.5)


print("Fit RF")
rf = RandomForestClassifier(n_estimators=20) 
rf.fit(train_fe.drop("Survived",axis=1),train_fe["Survived"])
rf_score = rf.score(valid_fe.drop("Survived",axis=1),valid_fe["Survived"])


print("Fit XGB")
bst = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators = 30,colsample_bytree=.8,importance="gain")
bst.fit(train_fe.drop("Survived",axis=1),train_fe["Survived"])
xgb_score = bst.score(valid_fe.drop("Survived",axis=1),valid_fe["Survived"])

print(f"XBART Hack: {xbart_hack_score}" )
print(f"XBART CLT: {xbart_clt_score}" )
print(f"XBART Probit: {xbart_probit_score}" )
print(f"RF: {rf_score}" )
print(f"XGB: {xgb_score}" )