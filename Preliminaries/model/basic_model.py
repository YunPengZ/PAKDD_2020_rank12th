import pandas as pd
import numpy as np
import os
import copy 
import gc
from tqdm import tqdm
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt
import math
from datetime import datetime
from sklearn.metrics import roc_auc_score
# import seaborn as sns
pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')
def get_unique_id(df):
    df['unique_id'] = df['serial_number'].map(str)+'_'+df['model'].map(str)
    return df
def time_to_first(df):
    df['time_to_first'] = (df['dt']-df['first_time']).dt.days
    return df

serial = pd.read_csv('../user_data/tmp_data/serial_v2.csv')
serial = serial[['serial_number','model','first_time']]
print('read serial end...')
df_1806 = pd.read_csv('../user_data/tmp_data/train_data/test_06.csv')
df_1806['dt'] = pd.to_datetime(df_1806['dt'])#,format='%Y%m%d')
df_1806 = df_1806.merge(serial,on=['serial_number','model'],how='left')
df_1806['first_time'] = pd.to_datetime(df_1806['first_time'])
df_1806 = time_to_first(df_1806)
print('read df_1806 end...')
test = pd.read_csv('../user_data/tmp_data/test_data/test_v3.csv')
test['dt'] = pd.to_datetime(test['dt'])#,format="%Y%m%d")
test = test.merge(serial,on=['serial_number','model'],how='left')
test['first_time'] = pd.to_datetime(test['first_time'])
test = time_to_first(test)
print('read test end..')
#训练集和验证集
#训练集和验证集
use_feas = [fea for fea in df_1806.columns if 'smart' in fea]
drop_cols = ['smart_3raw','smart_10raw','smart_240_normalized','smart_242_normalized','smart_241_normalized']
use_feas = [fea for fea in use_feas if fea not in drop_cols]
use_feas.append('time_to_first')

target = 'get_fault_in_30_days'
mean_feas = [fea for fea in use_feas if 'mean' in fea]
diff_feas = [fea for fea in use_feas if 'diff' in fea and 'smart_1_' not in fea]
old_feas = [fea for fea in use_feas if 'normalized_' not in fea and 'raw_' not in fea and 'smart_1_' not in fea and 'smart_1raw' not in fea]
use_feas = old_feas+diff_feas+mean_feas
print(len(use_feas))

##########训练模型
clf_res = LGBMClassifier(
    learning_rate=0.001,
    n_estimators=115,
    num_leaves=127,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=2019,
    is_unbalenced = 'True',
    metric=None
)
clf_res.fit(
        df_1806[use_feas], df_1806[target],
        eval_metric='auc',
        eval_set=[(df_1806[use_feas],df_1806[target])],
#         early_stopping_rounds=50,
        verbose=50
)

clf_pred = clf_res.predict_proba(test[use_feas])[:,1]
print('predict end...')
threshold = 0.00866
res = test.copy()
res['pred'] = clf_pred#RF+规则
res['manu'] = 'A'
res = res[['manu','model','serial_number','pred','dt']]
res = res[res.model==2]
res = res[res.pred>threshold]
res = res.sort_values('dt').drop_duplicates(['serial_number','model'])
res = res.drop('pred',axis=1)
res.to_csv('../prediction_result/predictions.csv',index=None, header=None,encoding = 'utf-8')