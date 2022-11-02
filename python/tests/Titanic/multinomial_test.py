import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score,log_loss
import xgboost as xgb
from xbart import XBART
from scipy.stats import norm

print("Read Data")
train_fe = pd.read_csv("train_fe.csv")
valid_fe = pd.read_csv("valid_fe.csv")

print("Fit XBART Multinomial")
model = XBART(num_trees=10, num_sweeps=30,n_min=10, 
		     tau=50/10, no_split_penalty = 1, burnin = 15, 
		     model="Multinomial",num_classes=2)
model.fit(train_fe.drop("Survived",axis=1),train_fe["Survived"])
preds_probit = model.predict(valid_fe.drop("Survived",axis=1),return_mean=False)
yhats = np.mean(preds_probit[:,15:,:],axis=1)[:,1]
print(log_loss(valid_fe['Survived'],yhats))

preds_probit = model.predict(valid_fe.drop("Survived",axis=1))
print(log_loss(valid_fe['Survived'],yhats))