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
from sklearn.externals import joblib

import warnings
import matplotlib.pyplot as plt
import math
from datetime import datetime
# import seaborn as sns
pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')


def get_df(file_dir,use_feas):
    chunk_size = 1e6
    print(file_dir)
    use_feas.extend(['serial_number','model','dt'])
    df = pd.read_csv(file_dir,usecols=use_feas, iterator=True)
    res_df = []
    while True:
        try:
            t = df.get_chunk(chunk_size)
            #na_t = t.isna().all()
            #inna_t = na_t[na_t == False]
            #print(len(inna_t))
            #res = t[inna_t.index]
            res_df.append(t)
        except:
            break
    df = pd.concat(res_df, ignore_index=True)
    del res_df
    try:
        df['dt'] = pd.to_datetime(df['dt'], format='%Y%m%d')
    except:
        df['dt'] = pd.to_datetime(df['dt'])
    df = reduce_mem(df)
    # df = df.sort_values('dt').drop_duplicates(['serial_number','model'])#所有磁盘保留第一次出现的记录
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


#读取使用的特征
import pickle
pkl_file = open('../user_data/tmp_data/diff_data.pkl', 'rb')
raw_feas,use_feas = pickle.load(pkl_file)

#直接读取模型 在文件夹下
clf = joblib.load('../user_data/model_data/model.dat')


###############################################
###处理测试集数据 测试集文件位置未知 按照提交时候的位置提交
window_list = []
for i in range(21):
    day = i + 11
    df_1808_file = './tcdata/disk_sample_smart_log_round2/disk_sample_smart_log_201808%02d_round2.csv' % day
    df_1808 = get_df(df_1808_file,raw_feas)
    #df_1808 = get_unique_id(df_1808)
    #df_1808_test = df_1808[df_1808.unique_id.isin(test_ids)]
    window_list.append(df_1808)
#窗口的初始大小是21 从11读到31
del df_1808
print('window initialied ends...')


def get_unique_id(df):
    df['unique_id'] = df['serial_number'].map(str) + '_' + df['model'].map(str)
    return df

def get_shift(old,cur,cur_date,raw_feas):
    df = pd.concat([old,cur]).reset_index(drop=True)
    df = df.sort_values('dt')
    for fea in raw_feas:
        df[fea+'_shift'] = df.groupby(['serial_number','model'])[fea].apply(lambda x :x.shift())
        df[fea] = df.groupby(['serial_number','model'])[fea].ffill()
        df[fea+'_shift'] = df.groupby(['serial_number','model'])[fea+'_shift'].ffill()
    df = df[df.dt==cur_date]
    return df

def construct_shift(old,cur,cur_date,feas):
    print('shape before processd',cur.shape)
    res = get_shift(old,cur,cur_date,feas)
    res = get_diff_rate(res,feas)
    print('shape after processed',res.shape)
    return res

def get_single_submit(val,pred,days = 15):
    sub = val[['serial_number', 'model', 'dt']].copy()
    print(sub.shape,'pred lens:',len(pred))
    sub['pred'] = pred
    sub = sub.sort_values('pred', ascending=False).head(days) #每天取前days个
    return sub

for i in range(30): # 30
    day = i + 1
    file_dir_test = './tcdata/disk_sample_smart_log_round2/disk_sample_smart_log_201809%02d_round2.csv' % day
    test_df = get_df(file_dir_test,raw_feas)
    ####向前取21天数据
    window_df = pd.concat(window_list,ignore_index=True)
    window_list.append(test_df)
    window_list = window_list[1:]#滑动窗口
    test_df = get_unique_id(test_df)
    test_ids = test_df['unique_id'].unique()
    window_df = get_unique_id(window_df)#不把get_unique_id放再外面 减少内存占用
    window_df = window_df[window_df.unique_id.isin(test_ids)]
    ##########合并
    cur_date = pd.datetime(2018,9,day)
    raw_feas = [fea for fea in raw_feas if 'smart' in fea]
    test = construct_shift(window_df,test_df,cur_date,raw_feas)
    test = process_time_diff(test,True)
    test_x = test[use_feas]
    del window_df,test_df#,old,cur
    ###直接覆盖掉这一轮的变量
    test_x = test_x.fillna(-1)
    submit = get_single_submit(test,clf.predict(test_x), days=19)#每天取top-k
    submit_list.append(submit)
    del submit,test_x
    gc.collect()

res = pd.concat(submit_list).reset_index(drop=True)
res['manufacturer'] = 'A'
res = res[['manufacturer', 'model', 'serial_number', 'dt','pred']]
res[['manufacturer', 'model', 'serial_number', 'dt']].to_csv("result.csv", index=False, header=None)
#结果打包
newZip = zipfile.ZipFile('result.zip','w')
newZip.write('result.csv',compress_type=zipfile.ZIP_DEFLATED)
newZip.close()
print('zip over !')