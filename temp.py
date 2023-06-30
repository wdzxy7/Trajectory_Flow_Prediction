import pandas as pd
from pandas import DataFrame


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


def change(x):
    x = str(x).replace('\'', '').replace(' ', '').replace('(', '').replace(')', '')
    return x


def spilt_data():
    data = pd.read_csv('./extract_train.csv')
    print(data.columns)
    data.columns = ['Time', 'Station', 'InNum', 'OutNum', 'train_count', 't']
    data['t'] = data['t'].apply(change)
    data['other_in'] = data['t'].map(lambda x: x.split(',')[0])
    data['other_out'] = data['t'].map(lambda x: x.split(',')[1])
    data['other_max_in'] = data['t'].map(lambda x: x.split(',')[2])
    data['other_min_in'] = data['t'].map(lambda x: x.split(',')[3])
    data['other_max_out'] = data['t'].map(lambda x: x.split(',')[4])
    data['other_min_out'] = data['t'].map(lambda x: x.split(',')[5])
    data['all_max_in'] = data['t'].map(lambda x: x.split(',')[6])
    data['all_min_in'] = data['t'].map(lambda x: x.split(',')[7])
    data['all_max_out'] = data['t'].map(lambda x: x.split(',')[8])
    data['all_min_out'] = data['t'].map(lambda x: x.split(',')[9])
    data['all_in'] = data['t'].map(lambda x: x.split(',')[10])
    data['all_out'] = data['t'].map(lambda x: x.split(',')[11])
    data.drop('t', axis=1, inplace=True)
    print(data.head())
    print(data.columns)
    data.to_csv('extract_train.csv', index=False)

spilt_data()