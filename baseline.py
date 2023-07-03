import datetime
import os
import re
import shutil
import paddle
import pickle
import warnings
import pandas as pd
from chinese_calendar import is_workday, is_in_lieu, is_holiday
from paddlets.models.model_loader import load
from paddlets.datasets.tsdataset import TSDataset
from paddlets.models.forecasting import LSTNetRegressor, TCNRegressor
from paddlets.transform import StandardScaler

warnings.filterwarnings('ignore')


def extract_station(x):
    station = re.compile("([A-Z])站").findall(x)
    if len(station)>0:
        return station[0]


def normal(dataset_train, dataset_val_test, dataset_val, dataset_test):
    scaler = StandardScaler()
    scaler.fit(dataset_train)
    dataset_train_scaled = scaler.transform(dataset_train)
    dataset_val_test_scaled = scaler.transform(dataset_val_test)
    dataset_val_scaled = scaler.transform(dataset_val)
    dataset_test_scaled = scaler.transform(dataset_test)
    return dataset_train_scaled, dataset_val_test_scaled, dataset_val_scaled, dataset_test_scaled, scaler


def read_data():
    df = pd.read_csv("data/客流量数据.csv")
    df["时间区段"] = df["时间区段"].apply(lambda x: x.split("-")[0])
    df["站点"] = df["站点"].apply(extract_station)
    df["时间"] = df["时间"].apply(lambda x: str(x)).str.cat(df['时间区段'], sep=" ")
    df["时间"] = pd.to_datetime(df["时间"])
    df = df.drop("时间区段", axis=1)
    df.columns = ["Time", "Station", "InNum", "OutNum"]
    return df


def count_train(x):
    station1 = ['A', 'B', 'C', 'D']
    station2 = ['E', 'F']
    if x in station1:
        return 3
    elif x in station2:
        return 2
    else:
        return 1


def count_other_in(x, data):
    station1 = {'A', 'B', 'C', 'D'}
    station2 = {'E', 'F'}
    other_in = other_out = other_max_in = other_min_in = other_max_out = other_min_out = all_max_in = all_min_in = all_max_out = all_min_out = all_in = all_out = 0
    t = x['Time']
    other_train = data.loc[(data.Time == t)]
    if x['Station'] in station1:
        try:
            station1 = station1 - {x['Station']}
            other_in = sum(other_train.loc[(other_train.Station.isin(station1))]['InNum'])
            other_out = sum(other_train.loc[(other_train.Station.isin(station1))]['OutNum'])
            other_max_in = max(other_train.loc[(other_train.Station.isin(station1))]['InNum'])
            other_min_in = min(other_train.loc[(other_train.Station.isin(station1))]['InNum'])
            other_max_out = max(other_train.loc[(other_train.Station.isin(station1))]['OutNum'])
            other_min_out = min(other_train.loc[(other_train.Station.isin(station1))]['OutNum'])
        except:
            pass
        station1 = station1 | {x['Station']}
        all_max_in = max(other_train.loc[(other_train.Station.isin(station1))]['InNum'])
        all_min_in = min(other_train.loc[(other_train.Station.isin(station1))]['InNum'])
        all_max_out = max(other_train.loc[(other_train.Station.isin(station1))]['OutNum'])
        all_min_out = min(other_train.loc[(other_train.Station.isin(station1))]['OutNum'])
        all_in = sum(other_train.loc[(other_train.Station.isin(station1))]['InNum'])
        all_out = sum(other_train.loc[(other_train.Station.isin(station1))]['OutNum'])
    elif x['Station'] in station2:
        station2 = station2 - {x['Station']}
        try:
            other_in = sum(other_train.loc[(other_train.Station.isin(station2))]['InNum'])
            other_out = sum(other_train.loc[(other_train.Station.isin(station2))]['OutNum'])
            other_max_in = max(other_train.loc[(other_train.Station.isin(station2))]['InNum'])
            other_min_in = min(other_train.loc[(other_train.Station.isin(station2))]['InNum'])
            other_max_out = max(other_train.loc[(other_train.Station.isin(station2))]['OutNum'])
            other_min_out = min(other_train.loc[(other_train.Station.isin(station2))]['OutNum'])
        except:
            pass
        station2 = station2 | {x['Station']}
        all_max_in = max(other_train.loc[(other_train.Station.isin(station2))]['InNum'])
        all_min_in = min(other_train.loc[(other_train.Station.isin(station2))]['InNum'])
        all_max_out = max(other_train.loc[(other_train.Station.isin(station2))]['OutNum'])
        all_min_out = min(other_train.loc[(other_train.Station.isin(station2))]['OutNum'])
        all_in = sum(other_train.loc[(other_train.Station.isin(station2))]['InNum'])
        all_out = sum(other_train.loc[(other_train.Station.isin(station2))]['OutNum'])
    else:
        all_max_in = x['InNum']
        all_min_in = x['InNum']
        all_max_out = x['OutNum']
        all_min_out = x['OutNum']
        all_in = x['InNum']
        all_out = x['OutNum']
    return other_in, other_out, other_max_in, other_min_in, other_max_out, other_min_out, all_max_in, all_min_in, all_max_out, all_min_out, all_in, all_out


def count_front(x, data):
    station_id = x['Station']
    t = x['Time']
    front_train = data.loc[(data.Station == station_id) & (data.Time == t - pd.Timedelta('15 minutes'))]
    try:
        return front_train['InNum'].values[0], front_train['OutNum'].values[0]
    except:
        return 0, 0


def count_front_other(x, data):
    station1 = {'A', 'B', 'C', 'D'}
    station2 = {'E', 'F'}
    other_in = 0
    other_out = 0
    t = x['Time']
    front_other_train = data.loc[(data.Time == t - pd.Timedelta('15 minutes'))]
    if x['Station'] in station1:
        try:
            station1 = station1 - {x['Station']}
            other_in = sum(front_other_train.loc[(front_other_train.Station.isin(station1))]['InNum'])
            other_out = sum(front_other_train.loc[(front_other_train.Station.isin(station1))]['OutNum'])
        except:
            pass
    elif x['Station'] in station2:
        station2 = station2 - {x['Station']}
        try:
            other_in = sum(front_other_train.loc[(front_other_train.Station.isin(station2))]['InNum'])
            other_out = sum(front_other_train.loc[(front_other_train.Station.isin(station2))]['OutNum'])
        except:
            pass
    else:
        try:
            other_in = front_other_train['InNum'].values[0]
            other_out = front_other_train['OutNum'].values[0]
        except:
            pass
    return other_in, other_out


def ptime_front_station_count(x, data):
    front_station = {'ABCD': [],
                     'EF': ['A', 'B', 'C', 'D'],
                     'G': ['E', 'F']}
    pre_station = x['Station']
    for key in front_station.keys():
        if pre_station in key:
            front_train = front_station[key]
            break
    t = x['Time']
    pre_in = sum(data.loc[(data.Time == t - pd.Timedelta('15 minutes')) & (data.Station.isin(front_train))]['InNum'])
    pre_out = sum(data.loc[(data.Time == t - pd.Timedelta('15 minutes')) & (data.Station.isin(front_train))]['OutNum'])
    return pre_in, pre_out


def ptime_back_station_count(x, data):
    front_station = {'ABCD': ['E', 'F'],
                     'EF': ['G'],
                     'G': []}
    pre_station = x['Station']
    for key in front_station.keys():
        if pre_station in key:
            next_train = front_station[key]
            break
    t = x['Time']
    next_in = sum(data.loc[(data.Time == t - pd.Timedelta('15 minutes')) & (data.Station.isin(next_train))]['InNum'])
    next_out = sum(data.loc[(data.Time == t - pd.Timedelta('15 minutes')) & (data.Station.isin(next_train))]['OutNum'])
    return next_in, next_out


def ntime_front_station_count(x, data):
    front_station = {'ABCD': [],
                     'EF': ['A', 'B', 'C', 'D'],
                     'G': ['E', 'F']}
    now_station = x['Station']
    for key in front_station.keys():
        if now_station in key:
            front_train = front_station[key]
            break
    t = x['Time']
    pre_in = sum(data.loc[(data.Time == t) & (data.Station.isin(front_train))]['InNum'])
    pre_out = sum(data.loc[(data.Time == t) & (data.Station.isin(front_train))]['OutNum'])
    return pre_in, pre_out


def ntime_back_station_count(x, data):
    front_station = {'ABCD': ['E', 'F'],
                     'EF': ['G'],
                     'G': []}
    now_station = x['Station']
    for key in front_station.keys():
        if now_station in key:
            next_train = front_station[key]
            break
    t = x['Time']
    next_in = sum(data.loc[(data.Time == t) & (data.Station.isin(next_train))]['InNum'])
    next_out = sum(data.loc[(data.Time == t) & (data.Station.isin(next_train))]['OutNum'])
    return next_in, next_out


def vec(x):
    vec_day = ''
    for i in range(x, 7):
        vec_day += '0'
    vec_day += '1'
    for i in range(x - 1):
        vec_day += '0'
    return vec_day


def station_vec(x):
    if x in ['A', 'B', 'C', 'D']:
        return '001'
    elif x == 'G':
        return '100'
    else:
        return '010'


def count_pre(x, data):
    front_station = {'ABCD': ['A', 'B', 'C', 'D'],
                     'EF': ['E', 'F'],
                     'G': ['G']}
    pre_station = x['Station']
    for key in front_station.keys():
        if pre_station in key:
            stations = front_station[key]
            break
    t = x['Time']
    next_in = sum(data.loc[(data.Time == t - pd.Timedelta('15 minutes')) & (data.Station.isin(stations))]['InNum'])
    next_out = sum(data.loc[(data.Time == t - pd.Timedelta('15 minutes')) & (data.Station.isin(stations))]['OutNum'])
    return next_in, next_out


def ntime_pbpre_station_count(x, data):
    front_station = {'ABCD': [],
                     'EF': [],
                     'G': ['A', 'B', 'C', 'D']}
    back_station = {'ABCD': ['G'],
                     'EF': [],
                     'G': []}
    now_station = x['Station']
    for key in front_station.keys():
        if now_station in key:
            front_train = front_station[key]
            back_train = back_station[key]
            break
    t = x['Time']
    pre_in = sum(data.loc[(data.Time == t) & (data.Station.isin(front_train))]['InNum'])
    pre_out = sum(data.loc[(data.Time == t) & (data.Station.isin(front_train))]['OutNum'])
    next_in = sum(data.loc[(data.Time == t) & (data.Station.isin(back_train))]['InNum'])
    next_out = sum(data.loc[(data.Time == t) & (data.Station.isin(back_train))]['OutNum'])
    return pre_in, pre_out, next_in, next_out


def count_pbpre(x, data):
    front_station = {'ABCD': [],
                     'EF': [],
                     'G': ['A', 'B', 'C', 'D']}
    back_station = {'ABCD': ['G'],
                    'EF': [],
                    'G': []}
    pre_station = x['Station']
    for key in front_station.keys():
        if pre_station in key:
            pre_stations = front_station[key]
            back_stations = back_station[key]
            break
    t = x['Time']
    pre_next_in = sum(data.loc[(data.Time == t - pd.Timedelta('30 minutes')) & (data.Station.isin(pre_stations))]['InNum'])
    pre_next_out = sum(data.loc[(data.Time == t - pd.Timedelta('30 minutes')) & (data.Station.isin(pre_stations))]['OutNum'])
    back_next_in = sum(data.loc[(data.Time == t - pd.Timedelta('30 minutes')) & (data.Station.isin(back_stations))]['InNum'])
    back_next_out = sum(data.loc[(data.Time == t - pd.Timedelta('30 minutes')) & (data.Station.isin(back_stations))]['OutNum'])
    return pre_next_in, pre_next_out, back_next_in, back_next_out


def count_next(x, data):
    station_id = x['Station']
    t = x['Time']
    front_train = data.loc[(data.Station == station_id) & (data.Time == t + pd.Timedelta('15 minutes'))]
    try:
        return front_train['InNum'].values[0], front_train['OutNum'].values[0]
    except:
        return 0, 0


def count_next_other(x, data):
    station1 = {'A', 'B', 'C', 'D'}
    station2 = {'E', 'F'}
    other_in = 0
    other_out = 0
    t = x['Time']
    front_other_train = data.loc[(data.Time == t + pd.Timedelta('15 minutes'))]
    if x['Station'] in station1:
        try:
            station1 = station1 - {x['Station']}
            other_in = sum(front_other_train.loc[(front_other_train.Station.isin(station1))]['InNum'])
            other_out = sum(front_other_train.loc[(front_other_train.Station.isin(station1))]['OutNum'])
        except:
            pass
    elif x['Station'] in station2:
        station2 = station2 - {x['Station']}
        try:
            other_in = sum(front_other_train.loc[(front_other_train.Station.isin(station2))]['InNum'])
            other_out = sum(front_other_train.loc[(front_other_train.Station.isin(station2))]['OutNum'])
        except:
            pass
    else:
        try:
            other_in = front_other_train['InNum'].values[0]
            other_out = front_other_train['OutNum'].values[0]
        except:
            pass
    return other_in, other_out


def nexttime_front_station_count(x, data):
    front_station = {'ABCD': [],
                     'EF': ['A', 'B', 'C', 'D'],
                     'G': ['E', 'F']}
    pre_station = x['Station']
    for key in front_station.keys():
        if pre_station in key:
            front_train = front_station[key]
            break
    t = x['Time']
    pre_in = sum(data.loc[(data.Time == t + pd.Timedelta('15 minutes')) & (data.Station.isin(front_train))]['InNum'])
    pre_out = sum(data.loc[(data.Time == t + pd.Timedelta('15 minutes')) & (data.Station.isin(front_train))]['OutNum'])
    return pre_in, pre_out


def nexttime_back_station_count(x, data):
    front_station = {'ABCD': ['E', 'F'],
                     'EF': ['G'],
                     'G': []}
    pre_station = x['Station']
    for key in front_station.keys():
        if pre_station in key:
            next_train = front_station[key]
            break
    t = x['Time']
    next_in = sum(data.loc[(data.Time == t + pd.Timedelta('15 minutes')) & (data.Station.isin(next_train))]['InNum'])
    next_out = sum(data.loc[(data.Time == t + pd.Timedelta('15 minutes')) & (data.Station.isin(next_train))]['OutNum'])
    return next_in, next_out


def count_nbpre(x, data):
    front_station = {'ABCD': [],
                     'EF': [],
                     'G': ['A', 'B', 'C', 'D']}
    back_station = {'ABCD': ['G'],
                    'EF': [],
                    'G': []}
    pre_station = x['Station']
    for key in front_station.keys():
        if pre_station in key:
            pre_stations = front_station[key]
            back_stations = back_station[key]
            break
    t = x['Time']
    pre_next_in = sum(data.loc[(data.Time == t + pd.Timedelta('30 minutes')) & (data.Station.isin(pre_stations))]['InNum'])
    pre_next_out = sum(data.loc[(data.Time == t + pd.Timedelta('30 minutes')) & (data.Station.isin(pre_stations))]['OutNum'])
    back_next_in = sum(data.loc[(data.Time == t + pd.Timedelta('30 minutes')) & (data.Station.isin(back_stations))]['InNum'])
    back_next_out = sum(data.loc[(data.Time == t + pd.Timedelta('30 minutes')) & (data.Station.isin(back_stations))]['OutNum'])
    return pre_next_in, pre_next_out, back_next_in, back_next_out


def count_next_all(x, data):
    front_station = {'ABCD': ['A', 'B', 'C', 'D'],
                     'EF': ['E', 'F'],
                     'G': ['G']}
    pre_station = x['Station']
    for key in front_station.keys():
        if pre_station in key:
            stations = front_station[key]
            break
    t = x['Time']
    next_in = sum(data.loc[(data.Time == t + pd.Timedelta('15 minutes')) & (data.Station.isin(stations))]['InNum'])
    next_out = sum(data.loc[(data.Time == t + pd.Timedelta('15 minutes')) & (data.Station.isin(stations))]['OutNum'])
    return next_in, next_out


def get_closeness(x, y, data):
    s = x['Station']
    t = x['Time']
    close = data.loc[(data.Station == s) & (data.Time == t - pd.Timedelta(str(y) + ' minutes'))]
    try:
        return close['InNum'].values[0], close['OutNum'].values[0]
    except:
        return 0, 0


def get_closeness2(x, y, data):
    front_station = {'ABCD': {'A', 'B', 'C', 'D'},
                     'EF': {'E', 'F'},
                     'G': {'G'}}
    now_station = x['Station']
    for key in front_station.keys():
        if now_station in key:
            all_station = front_station[key]
            break
    all_station = all_station - set(now_station)
    t = x['Time']
    close = data.loc[(data.Station.isin(all_station)) & (data.Time == t - pd.Timedelta(str(y) + ' minutes'))]
    try:
        return sum(close['InNum']), sum(close['OutNum'])
    except:
        return 0, 0


def get_period(x, y, data):
    s = x['Station']
    t = x['Time']
    close = data.loc[(data.Station == s) & (data.Time == t - pd.Timedelta(str(y) + ' days'))]
    try:
        return close['InNum'].values[0], close['OutNum'].values[0]
    except:
        return 0, 0


def get_period2(x, y, data):
    front_station = {'ABCD': {'A', 'B', 'C', 'D'},
                     'EF': {'E', 'F'},
                     'G': {'G'}}
    now_station = x['Station']
    for key in front_station.keys():
        if now_station in key:
            all_station = front_station[key]
            break
    all_station = all_station - set(now_station)
    t = x['Time']
    close = data.loc[(data.Station.isin(all_station)) & (data.Time == t - pd.Timedelta(str(y) + ' days'))]
    try:
        return sum(close['InNum']), sum(close['OutNum'])
    except:
        return 0, 0


def feature_extract(data):
    # pre
    data['Time'] = pd.to_datetime(data['Time'], errors='coerce')
    data['clock'] = data['Time'].dt.strftime('%H:%M:%S')
    '''
    # ----------------------------------列车本站相关特征-------------------------------------------------
    # 该站点有几个列车
    data['train_count'] = data["Station"].apply(lambda x: count_train(x))
    # 该站点其余列车上站，出站人数;
    # 该站其余列车进站最大人数,最小人数;
    # 该站其余列车出站最大人数,最小人数;
    # 该站进站最大人数,最小人数;
    # 该站出站最大人数,最小人数;
    # 该站进站总人数;
    # 该站出站总人数;
    data[['other_in', 'other_out', 'other_max_in', 'other_min_in', 'other_max_out', 'other_min_out',
         'all_max_in', 'all_min_in', 'all_max_out', 'all_min_out', 'all_in', 'all_out']] = \
        data.apply(lambda x: count_other_in(x, data), axis=1, result_type='expand')
    # 该站此时刻进出站人数差
    data['sub_now_all'] = data['all_in'] - data['all_out']
    # 该列车此时刻进出站人数差
    data['sub_now'] = data['InNum'] - data['OutNum']
    data['station_vec'] = data['Station'].apply(lambda x: station_vec(x))
    # ----------------------------------时间相关特征--------------------------------------------------
    # 是否为早高峰
    data['m_rush'] = data['clock'].apply(lambda x: 1 if pd.Timestamp("07:00:00") <= pd.Timestamp(x) <= pd.Timestamp("09:00:00") else 0)
    # 是否是中午下班时间段
    data['noon_off'] = data['clock'].apply(lambda x: 1 if pd.Timestamp("11:30:00") <= pd.Timestamp(x) <= pd.Timestamp("12:30:00") else 0)
    # 是否是下午上班时间段
    data['af_on'] = data['clock'].apply(lambda x: 1 if pd.Timestamp("13:30:00") <= pd.Timestamp(x) <= pd.Timestamp("14:30:00") else 0)
    # 是否是晚高峰
    data['night_rush'] = data['clock'].apply(lambda x: 1 if pd.Timestamp("17:30:00") <= pd.Timestamp(x) <= pd.Timestamp("19:30:00") else 0)
    # 是否是节假日
    data['is_holiday'] = data['Time'].apply(lambda x: 1 if is_holiday(x) else 0)
    # 是否是调休日
    data['is_change_work'] = data['Time'].apply(lambda x: 1 if is_in_lieu(x) else 0)
    # 是否是工作日
    data['is_work'] = data['Time'].apply(lambda x: 1 if is_workday(x) else 0)
    # 今天是周几
    data['date'] = data['Time'].dt.dayofweek + 1
    # 是否是周末
    data['week'] = data['date'].apply(lambda x: 1 if x > 5 else 0)
    # 周几向量化
    data['date'] = data['date'].apply(lambda x: vec(x))
    # 是否是休班时间
    data['stop'] = data['Time'].apply(lambda x: 1 if pd.Timestamp("01:45:00") < pd.Timestamp(x) < pd.Timestamp("05:00:00") else 0)
    # 是否是刚开始运行时间段
    data['open'] = data['Time'].apply(lambda x: 1 if pd.Timestamp("05:00:00") <= pd.Timestamp(x) <= pd.Timestamp("06:15:00") else 0)
    # 是否是临近收班时间段
    data['close'] = data['Time'].apply(lambda x: 1 if pd.Timestamp("10:45:00") <= pd.Timestamp(x) <= pd.Timestamp("01:45:00") else 0)
    # ----------------------------------列车当前时刻前后站相关特征---------------------------------------
    # 该站列车上一站进出站人数
    data[['pre_station_in', 'pre_station_out']] = data.apply(lambda x: ntime_front_station_count(x, data), axis=1, result_type='expand')
    # 该站列车上一站进出站人数差
    data['sub_pre_station'] = data['pre_station_in'] - data['pre_station_out']
    # 该站列车下一站进出站人数
    data[['next_station_in', 'next_station_out']] = data.apply(lambda x: ntime_back_station_count(x, data), axis=1, result_type='expand')
    # 该站列车下一站进出站人数差
    data['sub_next_station'] = data['next_station_in'] - data['next_station_out']
    # 该列上上站进出站人数;下下站进出站人数
    data[['ppre_station_in', 'ppre_station_out', 'nnext_station_in', 'nnext_station_out']] = data.apply(lambda x: ntime_pbpre_station_count(x, data), axis=1, result_type='expand')
    # 上上站进出站人数差
    data['sub_ppre'] = data['ppre_station_in'] - data['ppre_station_out']
    # 下下站进出站人数差
    data['sub_nnext'] = data['nnext_station_in'] - data['nnext_station_out']
    # ----------------------------------列车上一时刻相关特征--------------------------------------------
    # 该列车上一时刻进，出站人数
    data[['front_in', 'front_out']] = data.apply(lambda x: count_front(x, data), axis=1, result_type='expand')
    # 该列车上一时刻进出站差
    data['sub_front'] = data['front_in'] - data['front_out']
    # 该站点上一时刻其余车进，出站人数
    data[['front_other_in', 'front_other_out']] = data.apply(lambda x: count_front_other(x, data), axis=1, result_type='expand')
    # 该站点上一时刻其余车进，出站人数人数差
    data['sub_front_other'] = data['front_other_in'] - data['front_other_out']
    # 上一时刻前一站进出站人数
    data[['front_station_pin', 'front_station_pout']] = data.apply(lambda x: ptime_front_station_count(x, data), axis=1, result_type='expand')
    # 上一时刻前一站进出站人数差
    data['sub_front_station'] = data['front_station_pin'] - data['front_station_pout']
    # 上一时刻下一站进出人数
    data[['back_station_pin', 'back_station_pout']] = data.apply(lambda x: ptime_back_station_count(x, data), axis=1, result_type='expand')
    # 上一时刻下一站进出站人数差
    data['sub_back_station'] = data['back_station_pin'] - data['back_station_pout']
    # 该站点上一时刻进出站总人数
    data[['pre_all_in', 'pre_all_out']] = data.apply(lambda x: count_pre(x, data), axis=1, result_type='expand')
    # 该站点上一时刻进出站人数差
    data['sub_pre_all'] = data['pre_all_in'] - data['pre_all_out']
    # 上上时刻前前一个站进出站总人数; 上上时刻下下一个站进出站总人数
    data[['ppre_all_in', 'ppre_all_out', 'bpre_all_in', 'bpre_all_out']] = data.apply(lambda x: count_pbpre(x, data), axis=1, result_type='expand')
    # 上上时刻前前一个站进出站总人数差
    data['sub_ppre_all'] = data['ppre_all_in'] - data['ppre_all_out']
    # 上上时刻前前一个站进出站总人数差
    data['sub_bpre_all'] = data['bpre_all_in'] - data['bpre_all_out']
    # ----------------------------------列车下一时刻相关特征--------------------------------------------
    # 该列车下一时刻进，出站人数
    data[['next_in', 'next_out']] = data.apply(lambda x: count_next(x, data), axis=1, result_type='expand')
    # 该列车下一时刻进出站差
    data['sub_next'] = data['next_in'] - data['next_out']
    # 该站点下一时刻其余车进，出站人数
    data[['next_other_in', 'next_other_out']] = data.apply(lambda x: count_next_other(x, data), axis=1,result_type='expand')
    # 该站点下一时刻其余车进，出站人数人数差
    data['sub_next_other'] = data['next_other_in'] - data['next_other_out']
    # 下一时刻前一站进出站人数
    data[['front_station_nin', 'front_station_nout']] = data.apply(lambda x: nexttime_front_station_count(x, data), axis=1,result_type='expand')
    # 下一时刻前一站进出站人数差
    data['subn_pre_station'] = data['front_station_nin'] - data['front_station_nout']
    # 下一时刻下一站进出人数
    data[['back_station_nin', 'back_station_nout']] = data.apply(lambda x: nexttime_back_station_count(x, data), axis=1,result_type='expand')
    # 下一时刻下一站进出站人数差
    data['subn_back_station'] = data['back_station_nin'] - data['back_station_nout']
    # 下下时刻前前一个站进出站总人数; 下下时刻下下一个站进出站总人数
    data[['pnext_all_in', 'pnext_all_out', 'bnext_all_in', 'bnext_all_out']] = data.apply(lambda x: count_nbpre(x, data),axis=1, result_type='expand')
    # 下下时刻前前一个站进出站总人数差
    data['sub_pnext_all'] = data['pnext_all_in'] - data['pnext_all_out']
    # 下下时刻前前一个站进出站总人数差
    data['sub_bnext_all'] = data['bnext_all_in'] - data['bnext_all_out']
    # 该站点下一时刻进出站总人数
    data[['next_all_in', 'next_all_out']] = data.apply(lambda x: count_next_all(x, data), axis=1, result_type='expand')
    # 该站点下一时刻进出站人数差
    data['sub_next_all'] = data['next_all_in'] - data['next_all_out']
    '''
    # ----------------------------------closeness/period--------------------------------------------

    # 本列车
    data[['this_in_close1', 'this_out_close1']] = data.apply(lambda x, y=15: get_closeness(x, y, data), axis=1,
                                                             result_type='expand')
    data[['this_in_close2', 'this_out_close2']] = data.apply(lambda x, y=30: get_closeness(x, y, data), axis=1,
                                                             result_type='expand')
    data[['this_in_close3', 'this_out_close3']] = data.apply(lambda x, y=45: get_closeness(x, y, data), axis=1,
                                                             result_type='expand')
    data[['this_in_close4', 'this_out_close4']] = data.apply(lambda x, y=60: get_closeness(x, y, data), axis=1,
                                                             result_type='expand')
    data[['this_in_close5', 'this_out_close5']] = data.apply(lambda x, y=75: get_closeness(x, y, data), axis=1,
                                                             result_type='expand')
    data[['this_in_close6', 'this_out_close6']] = data.apply(lambda x, y=90: get_closeness(x, y, data), axis=1,
                                                             result_type='expand')
    data[['this_in_period1', 'this_out_period1']] = data.apply(lambda x, y=21: get_period(x, y, data), axis=1,
                                                             result_type='expand')
    data[['this_in_period2', 'this_out_period2']] = data.apply(lambda x, y=28: get_period(x, y, data), axis=1,
                                                               result_type='expand')
    data[['this_in_period3', 'this_out_period3']] = data.apply(lambda x, y=35: get_period(x, y, data), axis=1,
                                                               result_type='expand')
    data[['this_in_period4', 'this_out_period4']] = data.apply(lambda x, y=42: get_period(x, y, data), axis=1,
                                                               result_type='expand')
    data[['this_in_period5', 'this_out_period5']] = data.apply(lambda x, y=49: get_period(x, y, data), axis=1,
                                                               result_type='expand')
    # 本站其他列车
    data[['other_in_close1', 'other_out_close1']] = data.apply(lambda x, y=15: get_closeness2(x, y, data), axis=1,
                                                             result_type='expand')
    data[['other_in_close2', 'other_out_close2']] = data.apply(lambda x, y=30: get_closeness2(x, y, data), axis=1,
                                                             result_type='expand')
    data[['other_in_close3', 'other_out_close3']] = data.apply(lambda x, y=45: get_closeness2(x, y, data), axis=1,
                                                             result_type='expand')
    data[['other_in_close4', 'other_out_close4']] = data.apply(lambda x, y=60: get_closeness2(x, y, data), axis=1,
                                                             result_type='expand')
    data[['other_in_close5', 'other_out_close5']] = data.apply(lambda x, y=75: get_closeness2(x, y, data), axis=1,
                                                             result_type='expand')
    data[['other_in_close6', 'other_out_close6']] = data.apply(lambda x, y=90: get_closeness2(x, y, data), axis=1,
                                                             result_type='expand')
    data[['other_in_period1', 'other_out_period1']] = data.apply(lambda x, y=21: get_period2(x, y, data), axis=1,
                                                               result_type='expand')
    data[['other_in_period2', 'other_out_period2']] = data.apply(lambda x, y=28: get_period2(x, y, data), axis=1,
                                                               result_type='expand')
    data[['other_in_period3', 'other_out_period3']] = data.apply(lambda x, y=35: get_period2(x, y, data), axis=1,
                                                               result_type='expand')
    data[['other_in_period4', 'other_out_period4']] = data.apply(lambda x, y=42: get_period2(x, y, data), axis=1,
                                                                 result_type='expand')
    data[['other_in_period5', 'other_out_period5']] = data.apply(lambda x, y=49: get_period2(x, y, data), axis=1,
                                                                 result_type='expand')

    data.drop(['clock'], axis=1, inplace=True)

    return data


def data_prepare():
    # df = read_data()
    df = pd.read_csv('./extract_train.csv')
    df = feature_extract(df)
    df.to_csv('extract_train.csv', index=False)
    # df = pd.read_csv('./data/test.csv')
    # df = feature_extract(df)
    # df.to_csv('./data/test.csv', index=False)


def train():
    df = pd.read_csv('./extract_train.csv')
    dataset_df = df[df['Station'] == station]
    dataset_df.drop('Station', axis=1, inplace=True)
    dataset_df = TSDataset.load_from_dataframe(
        dataset_df,
        time_col='Time',
        target_cols=['InNum', 'OutNum'],
        freq='15min',
        fill_missing_dates=True,
        fillna_method='zero'
    )
    dataset_train, dataset_val_test = dataset_df.split("2023-02-05 23:45:00")
    dataset_val, dataset_test = dataset_val_test.split("2023-03-29 23:45:00")
    dataset_train_scaled, dataset_val_test_scaled, dataset_val_scaled, dataset_test_scaled, scaler = normal(
        dataset_train, dataset_val_test, dataset_val, dataset_test)
    paddle.seed(2023)
    model = LSTNetRegressor(
        in_chunk_len=4 * 24,
        out_chunk_len=4 * 24,
        max_epochs=200,
        patience=20,
        eval_metrics=["mae"],
        optimizer_fn=paddle.optimizer.adam.Adam,
        optimizer_params=dict(learning_rate=1e-4)
    )
    # model = TCNRegressor(in_chunk_len=4 * 24,
    #     out_chunk_len=4 * 24,
    #     max_epochs=200,
    #     patience=20,
    #     eval_metrics=["mae"],
    #     optimizer_fn=paddle.optimizer.adam.Adam,
    #     optimizer_params=dict(learning_rate=1e-4))
    model.fit(dataset_train_scaled, dataset_val_scaled)
    # save model
    model_save_path = os.path.join("models/lstm", station)
    shutil.rmtree(model_save_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    save_path = os.path.join(model_save_path, "model")
    model.save(save_path)
    pickle.dump(scaler, open('./scaler.pkl', 'wb'))


def predict():
    dataset_test_df = pd.read_csv("data/test.csv")

    dataset_test_df = dataset_test_df[dataset_test_df['Station'] == station]
    dataset_test_df = TSDataset.load_from_dataframe(
        dataset_test_df,
        time_col='Time',
        target_cols=['InNum', 'OutNum'],
        freq='15min',
        fill_missing_dates=False,
    )
    with open("./scaler.pkl", "rb") as r:
        scaler = pickle.load(r)
    dataset_test_scaled = scaler.transform(dataset_test_df)
    model = load("models/lstm/{}/model".format(station))
    res = model.predict(dataset_test_scaled)
    res_inversed = scaler.inverse_transform(res)
    res_inversed = res_inversed.to_dataframe(copy=True)
    res_inversed["InNum"] = res_inversed["InNum"].apply(lambda x: 0 if x < 0 else int(x))
    res_inversed["OutNum"] = res_inversed["OutNum"].apply(lambda x: 0 if x < 0 else int(x))
    res_inversed["Station"] = station  # 添加站点标识
    res_inversed.index.name = "Time"
    # 输出结果文件
    if not os.path.exists("./results"):
        os.makedirs("./results")
    res_inversed.to_csv("results/{}_result.csv".format(station))
    print(res_inversed.head())


def merge_res():
    df1 = pd.read_csv('./results/A_result.csv')
    df2 = pd.read_csv('./results/B_result.csv')
    df3 = pd.read_csv('./results/C_result.csv')
    df4 = pd.read_csv('./results/D_result.csv')
    df5 = pd.read_csv('./results/E_result.csv')
    df6 = pd.read_csv('./results/F_result.csv')
    df7 = pd.read_csv('./results/G_result.csv')
    df = df1.append(df2)
    df = df.append(df3)
    df = df.append(df4)
    df = df.append(df5)
    df = df.append(df6)
    df = df.append(df7)
    df.to_csv('./result.csv', index=False)


def main():
    train()
    predict()


def remove_no_train(t, c):
    if pd.Timestamp("01:45:00") < pd.Timestamp(t) < pd.Timestamp("05:00:00"):
        return 0
    else:
        return c


def remove_stop():
    df = pd.read_csv('./result.csv')
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df['clock'] = df['Time'].dt.strftime('%H:%M:%S')
    df['InNum'] = df.apply(lambda x: remove_no_train(x['clock'], x['InNum']), axis=1)
    df['OutNum'] = df.apply(lambda x: remove_no_train(x['clock'], x['OutNum']), axis=1)
    df.drop(['clock'], axis=1, inplace=True)
    df.to_csv('./result.csv', index=False)


if __name__ == '__main__':
    data_prepare()
    stations = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    for station in stations:
        main()
    merge_res()
    remove_stop()