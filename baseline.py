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
from paddlets.models.forecasting import LSTNetRegressor
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


def count_pre_station(x, data):
    front_station = {'ABCD': [],
                     'EF': ['A', 'B', 'C', 'D'],
                     'G': ['E', 'F']}
    now_station = x['Station']
    for key in front_station.keys():
        if now_station in key:
            front_train = front_station[key]
            break
    t = x['Time']
    pre_in = sum(data.loc[(data.Time == t - pd.Timedelta('15 minutes')) & (data.Station.isin(front_train))]['InNum'])
    pre_out = sum(data.loc[(data.Time == t - pd.Timedelta('15 minutes')) & (data.Station.isin(front_train))]['OutNum'])
    return pre_in, pre_out


def count_next_station(x, data):
    front_station = {'ABCD': ['E', 'F'],
                     'EF': ['G'],
                     'G': []}
    now_station = x['Station']
    for key in front_station.keys():
        if now_station in key:
            next_train = front_station[key]
            break
    t = x['Time']
    next_in = sum(data.loc[(data.Time == t - pd.Timedelta('15 minutes')) & (data.Station.isin(next_train))]['InNum'])
    next_out = sum(data.loc[(data.Time == t - pd.Timedelta('15 minutes')) & (data.Station.isin(next_train))]['OutNum'])
    return next_in, next_out


def feature_extract(data):
    # pre
    data['Time'] = pd.to_datetime(data['Time'], errors='coerce')
    data['clock'] = data['Time'].dt.strftime('%H:%M:%S')
    '''
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
    '''
    # 该列车上一时刻进，出站人数
    data[['front_in', 'front_out']] = data.apply(lambda x: count_front(x, data), axis=1, result_type='expand')
    # 该列车上一时刻进出站差
    data['sub_front'] = data['front_in'] - data['front_out']
    # 该站点上一时刻其余车进，出站人数
    data[['front_other_in', 'front_other_out']] = data.apply(lambda x: count_front_other(x, data), axis=1, result_type='expand')
    # 该站点上一时刻其余车进，出站人数人数差
    data['sub_front_other'] = data['front_other_in'] - data['front_other_out']
    # 该列车此时刻进出站人数差
    data['sub_now'] = data['InNum'] - data['OutNum']
    # 该站此时刻进出站人数差
    data['sub_now_all'] = data['all_in'] - data['all_out']
    # 该站列车上一站进出站人数
    data[['pre_station_in', 'pre_station_out']] = data.apply(lambda x: count_pre_station(x, data), axis=1, result_type='expand')
    # 该站列车上一站进出站人数差
    data['sub_pre_station'] = data['pre_station_in'] - data['pre_station_out']
    # 该站列车下一站进出站人数
    data[['next_station_in', 'next_station_out']] = data.apply(lambda x: count_next_station(x, data), axis=1, result_type='expand')
    # 该站列车下一站进出站人数差
    data['sub_next_station'] = data['next_station_in'] - data['next_station_out']

    return data


def data_prepare():
    # df = read_data()
    df = pd.read_csv('./extract_train.csv')
    df = feature_extract(df)
    df.to_csv('extract_train.csv', index=False)


def train():
    df = pd.read_csv('./extract_train.csv')
    dataset_df = df[df['Station'] == station]
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
        optimizer_params=dict(learning_rate=3e-3)
    )
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
    if station == 'A':
        data_prepare()
    train()
    predict()
    merge_res()


if __name__ == '__main__':
    stations = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    for station in stations:
        main()