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

def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        print(col)
        col_type = df[col].dtypes
        if col_type not in [object ,np.dtype('datetime64[ns]')] :
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


#获得变化率特征
def get_diff(df,feas):
    for fea in feas:
        df[fea+'_diff'] = df[fea]-df[fea+'_shift']
    return df
def get_diff_rate(df,feas):
    for fea in feas:
        df[fea+'_diff_rate'] = np.where(df[fea+'_shift']==0,None,np.abs(df[fea+'_diff']/df[fea+'_shift']))
    return df

def fill_na_shift(df,feas):
    df[feas] = df.groupby(['serial_number','model'])[feas].ffill()
    return df


def get_window_observation(df,window_sz = 7):#可以调整一下窗口大小 窗口小一些 说不定效果好些
#     fault = get_unique_id(fault)
#     fault_ids = fault['unique_id'].unique()
#     df = get_unique_id(df)
    bad_pan = df[df[target]==1]
    good_pan = df[df.tag==-1]
    observe_pan = df[df[target]==0][df.tag!=-1]#
    observe = observe_pan.sort_values('dt',ascending=False).groupby(['serial_number','model']).head(window_sz)
    observe_pan = observe_pan[~observe_pan.index.isin(observe.index)]#中立盘保留观察周期前的数据
    res = pd.concat([good_pan,observe_pan,bad_pan]).reset_index(drop=True).sort_values('dt')
    return res

#计算日均读入/读出数据量----smart_242+241/smart_9 总共的I/O量 +日均的I/O量
def process_time_diff(df,is_first=False):#启动后的平均运行时间
    if is_first:
        df['smart_9raw'] = df['smart_9raw']/24
    #记录重映射的个数 如果数据出错 这个数据就会被放大一次 因为5增加而197减少
    df['remap_cnt'] = df['smart_5raw']-df['smart_197raw']
    #把重映射的扇区和待映射的扇区加在一起 如果数目过大 可能会使硬盘接近崩溃 但是这样需要知道总的扇区个数
    df['bad_sector_cnt'] = df['smart_197raw']+df['smart_198raw']
    #统计磁盘I/O
    df['io_total'] = df['smart_241raw']+df['smart_242raw']
    df['io_avg'] = df['io_total']/df['smart_9raw']#raw_9的最小值是0.25 
    df['io_avg_start'] = df['io_total']/np.where(df['smart_12raw']==0,-df['io_total'],df['smart_12raw'])#每启动一次做多少io
    ### 12 表示开关机（....通电）次数 9是运行时间 两者相除表示 每次通电的使用时间
    df['time_per_start'] = df['smart_9raw']/np.where(df['smart_12raw']==0,-df['smart_9raw'],df['smart_12raw'])#如果底数为0 那么最终返回-1

#     df['io_total'] = np.log(df['io_total'])
    
    return df

def modify(df,feas):
    df[feas] = np.where(df[feas]>1,1,df[feas])
    return df


def fill_na(df,feas,is_val=False):
    if is_val:
        X_df = df[use_feas]
        y_df = df[target]
    else:
        model_1 = df[df.model==1][df.tag.isin([-1,0,1,2,3,4,5])]
        model_2 = df[df.model==2][df.tag.isin([-1,0,3,4])]
        X_df = pd.concat([model_1,model_2]).reset_index(drop=True)
        X_df = X_df.sort_values('dt')
        y_df = X_df[target]
        X_df = X_df[use_feas]
    X_df = X_df.fillna(-1)
    return X_df,y_df

fault = pd.read_csv('../user_data/tmp_data/disk_sample_fault_tag.csv')
fault['fault_time'] = pd.to_datetime(fault['fault_time'])
fault_tag_08 = pd.read_csv('./second_part/disk_sample_fault_tag_201808.csv')
fault_tag_08['fault_time'] = pd.to_datetime(fault_tag_08['fault_time'])
fault = pd.concat([fault,fault_tag_08])



df_1807 = get_df('./data/round1_train/disk_sample_smart_log_201807.csv')
df_1805 = get_df('./data/round1_train/disk_sample_smart_log_201805.csv')
df_1806 = get_df('./data/round1_train/disk_sample_smart_log_201806.csv')
df_1803 = get_df('./data/round1_train/disk_sample_smart_log_201803.csv')
df_1804 = get_df('./data/round1_train/disk_sample_smart_log_201804.csv')


df_1807['dt'] = pd.to_datetime(df_1807['dt'],format='%Y%m%d')
df_1806['dt'] = pd.to_datetime(df_1806['dt'],format='%Y%m%d')
df_1805['dt'] = pd.to_datetime(df_1805['dt'],format='%Y%m%d')
df_1804['dt'] = pd.to_datetime(df_1804['dt'],format='%Y%m%d')
df_1803['dt'] = pd.to_datetime(df_1803['dt'],format='%Y%m%d')


df_1806 = reduce_mem(df_1806)
df_1807 = reduce_mem(df_1807)
df_1805 = reduce_mem(df_1805)
df_1804 = reduce_mem(df_1804)
df_1803 = reduce_mem(df_1803)

val_07 = pd.concat([df_1806,df_1807])
test_train = pd.concat([df_1803,df_1804,df_1805])

drop_cols = ['smart_3raw','smart_10raw','smart_240_normalized','smart_242_normalized','smart_241_normalized']
test_train = test_train.drop(drop_cols,axis=1)
val_07 = val_07.drop(drop_cols,axis=1)

test_train = get_label(test_train,fault)
val_07 = get_label(val_07,fault)

feas = [fea for fea in test_train.columns if 'smart' in fea]
###获得shift特征

for fea in feas:
    test_train[fea+'_shift'] = test_train.groupby(['serial_number','model'])[fea].apply(lambda x :x.shift())
    val_07[fea+'_shift'] = val_07.groupby(['serial_number','model'])[fea].apply(lambda x :x.shift())
    
target = 'get_fault_in_30_days'

train = test_train
val = val_07
del test_train,val_07


use_feas = [fea for fea in train.columns if 'smart' in fea]
train = fill_na_shift(train,use_feas)
val = fill_na_shift(val,use_feas)

feas = [fea for fea in use_feas if 'shift' not in fea]

train = get_diff(train,feas)
val = get_diff(val,feas)
train = get_diff_rate(train,feas)
val = get_diff_rate(val,feas)

train['dt'] = pd.to_datetime(train['dt'])
val['dt'] = pd.to_datetime(val['dt'])

t = get_window_observation(train,30)
tmp_train = train
train =t

#获得一些I/O特征
train = process_time_diff(train,True)
val = process_time_diff(val,True)


diff_rate_feas = [fea for fea in train.columns if 'diff_rate' in fea]

train = modify(train,diff_rate_feas)
val = modify(val,diff_rate_feas)

#读取使用的特征
import pickle
pkl_file = open('../user_data/tmp_data/diff_data.pkl', 'rb')
feas,use_feas = pickle.load(pkl_file)


X_train,y_train = fill_na(train,use_feas,False)
X_val,y_val = fill_na(val,use_feas,True)


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
    'category_feature':['model'],
    'n_jobs': -1
}

#训练
clf = lgb.train(params=lgb_params, train_set=dtrain, valid_sets=[dtrain, d_val],num_boost_round=100)
joblib.dump(clf, '../user_data/model_data/model.dat')#