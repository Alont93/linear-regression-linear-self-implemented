import numpy as np
import pandas as pd


DATA_PATH = './kc_house_data.csv'

def np_read_data():
    return np.genfromtxt(DATA_PATH, delimiter=',',skip_header=1)

def read_data():
    data_table = pd.read_csv(DATA_PATH)
    data_mat = data_table.values[:, 2:]
    return data_mat.astype(np.float64)

def split_featurs_and_lables(data):
    return (data[:,1:], data[:,0])

data = read_data()
X, Y = split_featurs_and_lables(data)

x =1