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

###模型使用lightgbm的rf模式 基本未调参
dtrain = lgb.Dataset(X_train,y_train, free_raw_data=False)
d_val = lgb.Dataset(X_val,y_val)
lgb_params = {
    'objective': 'regression_l2',
    'metric':'auc',
    'boosting_type': 'rf',
#     ''，
    'subsample': 0.623,
    'colsample_bytree': 0.7,
    'num_leaves': 127,
    'max_depth': 8,#明显由欠拟合转为过拟合
    'seed': 2019,
    'bagging_freq': 1,
    'early_stopping_rounds':100,
    'category_feature':['model'],
    'n_jobs': -1
}

#训练
clf = lgb.train(params=lgb_params, train_set=dtrain, valid_sets=[dtrain, d_val],num_boost_round=100)