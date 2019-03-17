import numpy as np
import pandas as pd
import datetime
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import mean_squared_error,log_loss
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')
np.random.seed(4590)

from XBART import XBART

print("Reading Data...")
path = "/Users/saaryalov/Kaggle/ELO/saar_model/Data_2/"

# df_train = pd.read_csv( path+ 'train_ok2_xbart_4.csv')
# df_test = pd.read_csv(path+ 'test_ok2_xbart_4.csv')

df_train = pd.read_csv( path+ 'train_ok2_xbart_norm_1.csv')
df_test = pd.read_csv(path+ 'test_ok2_xbart_norm_1.csv')

print(df_train.shape)
print(df_test.shape)


print('Preprocess...')
target = df_train['target']
del df_train['target']; 

print(target.shape)
#del df_train['Unnamed: 0']; del df_test['Unnamed: 0'];
features = [c for c in df_train.columns if c not in ['card_id', 'first_active_month','outliers']]
cont = ["new_hist_purchase_amount_var",
"new_hist_purchase_amount_mean",
"hist_month_lag_var",
"new_hist_purchase_date_max",
"hist_purchase_date_max",
"hist_purchase_date_min",
"hist_purchase_amount_sum",
"hist_purchase_amount_var",
"hist_purchase_amount_mean" ]

cat= ['new_hist_purchase_date_uptonow',
 'hist_month_diff_mean',
 'hist_category_1_sum',
 'new_hist_category_1_sum',
 'new_hist_purchase_amount_max',
 'hist_merchant_id_nunique',
 'new_hist_month_lag_mean',
 'elapsed_time',
 'new_hist_installments_mean',
 'new_hist_month_diff_mean',
 'hist_category_1_mean',
 'new_hist_card_id_size',
 'new_hist_purchase_date_diff',
 'hist_purchase_date_uptonow',
 'hist_month_nunique',
 'hist_authorized_flag_mean',
 'feature_1',
 'hist_installments_sum',
 'new_hist_merchant_category_id_nunique',
 'hist_first_buy',
 'hist_month_lag_mean',
 'hist_purchase_amount_min',
 'hist_category_3_mean_mean',
 'hist_weekofyear_nunique',
 'hist_purchase_date_average',
 'new_hist_purchase_date_average',
 'hist_subsector_id_nunique',
 'new_hist_purchase_amount_min',
 'new_hist_month_lag_var',
 'new_hist_installments_var',
 'hist_installments_mean',
 'hist_purchase_date_diff',
 'hist_category_2_mean_mean',
 'new_hist_category_3_mean_mean',
 'hist_merchant_category_id_nunique']
features = cont+ cat

print(features)
print("Define XBART Model")
m = 20
tau = .67*np.var(target)/m
params = OrderedDict([('M',m),('L',1),("N_sweeps",250)
							,("Nmin",1),("Ncutpoints",30)
							,("alpha",0.95),("beta",1.75),("tau",tau),
							("burnin",15),("mtry",8),("max_depth_num",30),
							("draw_sigma",False),("kap",16),("s",4),("verbose",True),
							("m_update_sigma",False),
							("draw_mu",False),("parallel",False)])



print("CV")
folds = KFold(n_splits=5, shuffle=True, random_state=2333)
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))


xbart = XBART(params)
# xbart.fit_2d_all(df_train[features].values,target.values,len(cat))
# predictions += xbart.predict_2d_all(df_test[features].values)[:,params["burnin"]:].mean(axis=1)
for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train)):
    print("fold {}".format(fold_))
    # trn_data = df_train.iloc[trn_idx][features], label=target.iloc[trn_idx])#, categorical_feature=categorical_feats)
    # val_data = lgb.Dataset(df_train.iloc[val_idx][features], label=target.iloc[val_idx])#, categorical_feature=categorical_feats)
    print("Fitting")
    xbart.fit_2d_all(df_train.iloc[trn_idx][features].values,target.iloc[trn_idx].values,0)

    print("Get train")
    y_pred_train = xbart.predict_2d_all(df_train.iloc[trn_idx][features].values)[:,params["burnin"]:].mean(axis=1)
    print("Fold {} has score: {:<8.5f}".format(fold_,mean_squared_error(y_pred_train, target.iloc[trn_idx].values)**0.5))

    print("Pred oof")
    y_pred_oof = xbart.predict_2d_all(df_train.iloc[val_idx][features].values)[:,params["burnin"]:].mean(axis=1)
    print("Fold {} has score: {:<8.5f}".format(fold_,mean_squared_error(y_pred_oof, target.iloc[val_idx].values)**0.5))
    oof[val_idx] = y_pred_oof/folds.n_splits
    print("Pred test")
    predictions += xbart.predict_2d_all(df_test[features].values)[:,params["burnin"]:].mean(axis=1)/ folds.n_splits

predictions_df = pd.concat([df_test["card_id"],pd.Series(predictions)],axis=1)
predictions_df.to_csv("xbart_pred_3.csv",index=False)
oof.to_csv("xbart_off_3.csv",index=False)
print("CV score: {:<8.5f}".format(mean_squared_error(oof, target.values)**0.5))

