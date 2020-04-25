import pandas as pd
import numpy as np
import os
import copy 
import gc
from tqdm import tqdm
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt
import math
from datetime import datetime
# import seaborn as sns
pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')

def get_df(file_dir):
    chunk_size = 1e6
    df = pd.read_csv(file_dir,iterator = True)
    res_df = []
    while True:
        try:
            t = df.get_chunk(chunk_size)
            na_t = t.isna().all()
            inna_t = na_t[na_t==False]
            res = t[inna_t.index]
            res_df.append(res)
        except:
            print('once end')
            break
    df = pd.concat(res_df,ignore_index = True)
    del res_df
    gc.collect()
    return df
###############################先看是否有label 
def get_label(df,fault):
    t = pd.merge(df,fault,on=['serial_number','model'],how='left')
    t['tag'] = t['tag'].fillna(-1).astype(int)
    dtime = pd.to_datetime(t['dt'],format='%Y%m%d')
    fault_time= pd.to_datetime(t['fault_time'],format='%Y-%m-%d')
    faut_time_delta_days = (fault_time-dtime).dt.days
    t['get_fault_in_30_days'] = np.where(faut_time_delta_days<=30,1,0)#需要大于0 么 如果是-1 即
    return t

def get_unique_id(df):
    df['unique_id'] = df['serial_number'].map(str)+'_'+df['model'].map(str)
    return df

def get_diff_rate(df,feature_aug):
    for fea in feature_aug:
        print(fea)
        df[fea+'_diff_shift'] = df.groupby(['serial_number','model'])[fea+'_diff'].shift(1,axis=0)
        df[fea+'_diff_rate'] = df[fea+'_diff']/df[fea+'_diff_shift']
    return df

def agg_features(df,col,agg_params,suffix=''):
    agg_dict = {}
    for agg in agg_params:
        agg_dict[col+'_'+agg+suffix] = agg
    t = df.groupby(['serial_number','model'])[col].agg(agg_dict).reset_index()
    return t

def get_kind_feature(old,cur_date,res,feature,agg_params,absolute=False):#获得每一类特征的一天，三天，一个月内的数据
    old_recent_1_days = old[old.dt==cur_date-pd.Timedelta(days=1)]
    old_recent_3_days = old[old.dt>=cur_date- pd.Timedelta(days=3)]
    old_recent_5_days = old[old.dt>=cur_date- pd.Timedelta(days=5)]
    old_recent_7_days = old[old.dt>=cur_date- pd.Timedelta(days=7)]
    old_recent_15_days = old[old.dt>=cur_date- pd.Timedelta(days=15)]
#     old_recent_month = old[old.dt>=cur_date- pd.Timedelta(days=30)]
###############################################################################平均值删掉
    if absolute:
        res[feature] = np.abs(res[feature])
    t = agg_features(old,feature,agg_params)
    res = pd.merge(res,t,on=['serial_number','model'],how='left')#res是group by 后的值
    t = old_recent_1_days.groupby(['serial_number','model'])[feature].agg({feature+'_1_days':'mean'})#可能会有多值
    res = pd.merge(res,t,on=['serial_number','model'],how='left')
    ###################################################3333
    t = agg_features(old_recent_3_days,feature,agg_params,'_3_days')
    res = pd.merge(res,t,on=['serial_number','model'],how='left')
    t = agg_features(old_recent_5_days,feature,agg_params,'_5_days')    
    res = pd.merge(res,t,on=['serial_number','model'],how='left')
        ##########################################7
    t = agg_features(old_recent_7_days,feature,agg_params,'_7_days')    
    res = pd.merge(res,t,on=['serial_number','model'],how='left')
    t = agg_features(old_recent_15_days,feature,agg_params,'_15_days')
    res = pd.merge(res,t,on=['serial_number','model'],how='left')
#     t = agg_features(old_recent_month,feature,agg_params,'_recent_month')
#     res = pd.merge(res,t,on=['serial_number','model'],how='left')
    return res

def get_diff_rate(df,col,absolute=False):
    df[col+'_1_days_diff'] = df[col]-df[col+'_1_days']
    df[col+'_3_days_diff'] = df[col]-df[col+'_mean_3_days']
    df[col+'_5_days_diff'] = df[col]-df[col+'_mean_5_days']
    df[col+'_7_days_diff'] = df[col]-df[col+'_mean_7_days']
    df[col+'_15_days_diff'] = df[col]-df[col+'_mean_15_days']
    df[col+'_recent_month_diff'] = df[col]-df[col+'_mean']
    if absolute==True:
        for suffix in ['_1_days_diff','_5_days_diff','_15_days_diff','_recent_month_diff']:
            df[col+suffix] = np.abs(df[col+suffix])
#     df[col+'_one_day_diff_rate'] = df[col+'_1_days_diff']/df[col+'_1_days']  #相较于上一天
    return df


#构造30天的训练集 和验证集
def construct_train(old,cur,cur_date,test=False):
    drop_features = ['dt','fault_time','tag','manufacturer','unique_id','diff_day','get_fault_in_30_days','manufacturer_x','manufacturer_y']
    keep_features = [fea for fea in test_train.columns if fea not in drop_features+drop_cols]
    if test==False:
        keep_features.append('get_fault_in_30_days')
    res = pd.DataFrame(cur[keep_features])#现有的数据  应该是现有数据和历史数据结合 现有数据全部保留 所以这样写 问题不大
    #####################################
    #统计历史数据 全部历史的统计量
    ############### smart_1 raw
    res = get_kind_feature(old,cur_date,res,'smart_1_normalized',['mean'])
    ############### smart_1 raw
    res = get_kind_feature(old,cur_date,res,'smart_1raw',['mean'])
    #################smart 3 normalized
    res = get_kind_feature(old,cur_date,res,'smart_3_normalized',['max','mean'])
    #################smart 4 normalized
#     res = get_kind_feature(old,cur_date,res,'smart_4_normalized',['max'])
    #################smart 4 normalized
#     res = get_kind_feature(old,cur_date,res,'smart_4raw',['max'])
    #####################################smart 5
    #5是一直在增长的 转化为normalize之后应该是一直在下降的 但是只要下降趋势是稳定的就可以 下降趋势
    #下降趋势应该怎么表示呢 30天变化值 除以3天的平均值 
    res = get_kind_feature(old,cur_date,res,'smart_5_normalized',['mean'])
    res = get_kind_feature(old,cur_date,res,'smart_5raw',['mean'])
    ################### smart_7
    res = get_kind_feature(old,cur_date,res,'smart_7raw',['mean'])
    res = get_kind_feature(old,cur_date,res,'smart_7_normalized',['mean'])
    ################### smart_10
    res = get_kind_feature(old,cur_date,res,'smart_10_normalized',['mean'])
    ################### smart_184 出厂时的坏盘记录 这个应该对磁盘的不同日期取值都一样吧
    ### 187 数据不为零（突然增大 则为异常点）
    res = get_kind_feature(old,cur_date,res,'smart_187raw',['mean'])
    ################### smart_188
#     res = get_kind_feature(old,cur_date,res,'smart_188raw',['max'])
#     res = get_kind_feature(old,cur_date,res,'smart_188_normalized',['max'])
     ################### smart_191
#     res = get_kind_feature(old,cur_date,res,'smart_191raw',['mean','max'])
    ################### smart_193
    res = get_kind_feature(old,cur_date,res,'smart_193raw',['mean'])
    ################### smart_192
    res = get_kind_feature(old,cur_date,res,'smart_192_normalized',['mean'])
    ################### smart_194
    res = get_kind_feature(old,cur_date,res,'smart_194_normalized',['mean','max'])
    ################### smart_195
    res = get_kind_feature(old,cur_date,res,'smart_195_normalized',['mean','max'])
    ################### smart_197
    res = get_kind_feature(old,cur_date,res,'smart_197_normalized',['mean','max'])
    res = get_kind_feature(old,cur_date,res,'smart_197raw',['mean','max'])

    ################### smart_198
    res = get_kind_feature(old,cur_date,res,'smart_198_normalized',['mean'])
    ################### smart_199
    res = get_kind_feature(old,cur_date,res,'smart_199raw',['mean','max'])

#     res = get_kind_feature(old,cur_date,res,'smart_199_normalized',['max'])
    ################### smart_240
    res = get_kind_feature(old,cur_date,res,'smart_240raw',['mean'])
    ################### smart_242
    res = get_kind_feature(old,cur_date,res,'smart_242raw',['mean'])
    ####################################
    #添加变化列和统计变化幅度
    res = get_diff_rate(res,'smart_1_normalized')#变化幅度和变化幅度除以原始的值 变化的值在与一周内比较 必须使用平均值
#     res = get_diff_rate(res,'smart_3_normalized')
    res = get_diff_rate(res,'smart_5_normalized')
    res = get_diff_rate(res,'smart_7raw')
#     res = get_diff_rate(res,'smart_10_normalized')
    res = get_diff_rate(res,'smart_187raw')
    res = get_diff_rate(res,'smart_192_normalized')
    res = get_diff_rate(res,'smart_194_normalized')
    res = get_diff_rate(res,'smart_195_normalized')
    res = get_diff_rate(res,'smart_197_normalized')
    #变化量除以原始的均值 表示变换量
    return res

import time
def merge_history(df,date_range,isTest=False):#对历史数据的处理 比如对5月2号 历史数据为5月1号数据加上之前的数据
    df_list = []
    for index,cur_date in enumerate(date_range):
        #31开始是不用的文件
        index = index + 31
        print('day',cur_date,index)
        start = time.time()
        #取五月的数据做训练
        #############################
        #历史数据 
        old = df[df.dt<cur_date][df.dt>=cur_date- pd.Timedelta(days=30)]
        ###############################
        #取当天的成绩
        cur = df[df.dt==cur_date]
        ##############################
        #两者合并
        print('before merged shape',old.shape)
        construct_df = construct_train(old,cur,cur_date,test=isTest)
        construct_df['dt'] =cur_date
        print('after merged shape',construct_df.shape)
        end = time.time()
        print(end-start)
        df_list.append(construct_df)
    history = pd.concat(df_list).reset_index(drop=True)
    return history

test = pd.read_csv('../data/disk_sample_smart_log_test_b.csv')
test = test.sort_values('dt')
print('test b read end')

fault = pd.read_csv('../data/disk_sample_fault_tag.csv')
df_1807 = get_df('../data/round1_train/disk_sample_smart_log_201807.csv')
df_1805 = get_df('../data/round1_train/disk_sample_smart_log_201805.csv')
df_1806 = get_df('../data/round1_train/disk_sample_smart_log_201806.csv')

test['dt'] = pd.to_datetime(test['dt'],format='%Y%m%d')
df_1807['dt'] = pd.to_datetime(df_1807['dt'],format='%Y%m%d')
df_1806['dt'] = pd.to_datetime(df_1806['dt'],format='%Y%m%d')
df_1805['dt'] = pd.to_datetime(df_1805['dt'],format='%Y%m%d')

test_train = pd.concat([df_1805,df_1806])
drop_cols = ['smart_3raw','smart_10raw','smart_240_normalized','smart_242_normalized','smart_241_normalized']
test_train = test_train.drop(drop_cols,axis=1)
test_train = get_label(test_train,fault)

test_train = merge_history(test_train,pd.date_range(start='20180601',end='20180630'))
test_train.to_csv('../user_data/tmp_data/train_data/test_06.csv')

test = get_unique_id(test)
test_ids = test['unique_id'].unique()
df_1807 = get_unique_id(df_1807)
df_1807_test = df_1807[df_1807.unique_id.isin(test_ids)]
test_df = pd.concat([df_1807_test,test]).reset_index(drop=True)

res_df = merge_history(test_df,pd.date_range(start='20180801',end='20180831'),True)
res_df.to_csv('../user_data/tmp_data/test_data/test_v3.csv')