import numpy as np
import pandas as pd
from xbart import XBART
import time 

### Helper Functions
def read_data(path,target="SalePrice"):
    df = pd.read_csv(path)
    return df.drop(target,axis=1).values,df[target].values

def rmse(target,pred):
    return np.sqrt(np.mean((target-pred)**2))

path = "/Users/saaryalov/Desktop/Thesis/python_example/"

# Read Data
train_fe,target_train = read_data(path+"train_fe.csv")
test_fe,target_test = read_data(path+"test_fe.csv")

# Define Result
result = pd.DataFrame(columns = ["RMSE","Time"])



# Create Object
# Define Params - Defined as a dictionary 
xbart_params = {"M":125, # Number of Trees
                "N_sweeps":60, # Number of MCMC loop
                "burnin":15, # Burn in Period
                "Ncutpoints":30, # 30 recursive cutpoints
                "mtry":7, # At each node, 7 possible variables
                "max_depth_num":25, # Max Depth
                "alpha":0.95,"beta":1.25} # Split prior values
xbart_params["tau"] = 1/xbart_params["M"] # Variance prior

# Create Object
xbart = XBART(xbart_params) # Contructor

# Fit 
time_1 = time.time()
xbart.fit(train_fe,target_train)
result.loc['XBART_cont','Time'] = time.time() - time_1

# Predict
y_pred = xbart.predict(test_fe)

# Result
y_hat_xbart = y_pred[:,xbart_params["burnin"]:].mean(axis=1)
result.loc['XBART_cont','RMSE'] = rmse(y_hat_xbart,target_test)

time_1 = time.time()
xbart.fit(train_fe,target_train,p_cat=29)
result.loc['XBART_mixed','Time'] = time.time() - time_1

# Result
y_pred = xbart.predict(test_fe)
y_hat_xbart = y_pred[:,xbart_params["burnin"]:].mean(axis=1)
result.loc['XBART_mixed','RMSE'] = rmse(y_hat_xbart,target_test)

print(result)

