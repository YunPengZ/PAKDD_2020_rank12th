import pandas as pd
import numpy as np
import os
import copy 
import gc
from tqdm import tqdm
import lightgbm as lgb
# from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt
import math
from datetime import datetime
# import seaborn as sns
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')

def get_first_time_df(file_dir):
    chunk_size = 1e6
    print(file_dir)
    df = pd.read_csv(file_dir,iterator = True)
    res_df = []
    while True:
        try:
            t = df.get_chunk(chunk_size)
            na_t = t.isna().all()
            inna_t = na_t[na_t==False]
            print(len(inna_t))
            res = t[inna_t.index]
            res_df.append(res)
        except:
            break
    df = pd.concat(res_df,ignore_index = True)
    df = df.sort_values('dt').drop_duplicates(['serial_number','model'])#所有磁盘保留第一次出现的记录
    return df

###需要data数据中有全部的数据集
###需要data数据中有全部的数据集
df_list = []
for i in range(7):
    month = i+1
    file_dir = '../data/round1_train/disk_sample_smart_log_2018%02d.csv' % month
    df_list.append(get_first_time_df(file_dir))
for i in range(6):
    month = i+7
    file_dir = '../data/round1_train/disk_sample_smart_log_2017%02d.csv' % month#分别处理2017年和2018年的数据
    df_list.append(get_first_time_df(file_dir))
file_dir_test_a = '../data/round1_testA/disk_sample_smart_log_test_a.csv'
file_dir_test_b = '../data/round1_testB/disk_sample_smart_log_test_b.csv'
df_list.append(get_first_time_df(file_dir_test_a))
df_list.append(get_first_time_df(file_dir_test_b))

res = pd.concat(df_list).reset_index(drop=True)
res = res.sort_values('dt').drop_duplicates(['serial_number','model'])
res['first_time'] = pd.to_datetime(res['dt'],format='%Y%m%d')
res = res[['serial_number','model','first_time']]
res.to_csv('../user_data/tmp_data/serial_v2.csv',index=None, header=None,encoding = 'utf-8')