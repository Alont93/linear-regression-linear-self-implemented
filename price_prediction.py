import numpy as np
import pandas as pd

DATE_LENGTH = 8
LABEL_COLUMN = 2
DATA_PATH = './kc_house_data.csv'

def np_read_data():
    return np.genfromtxt(DATA_PATH, delimiter=',',skip_header=1)

def read_data():
    data_table = pd.read_csv(DATA_PATH)
    fix_dates(data_table)
    data_table = hadle_categorical_variables(data_table)
    data_mat = data_table.values
    return data_mat.astype(np.float64)

def hadle_categorical_variables(data_table):
    data_table = pd.concat((data_table,pd.get_dummies(data_table.zipcode)),1)
    # remove linear indipendant variable
    data_table.drop(data_table.columns[[-1,]], axis=1, inplace=True)
    return data_table

def fix_dates(data_table):
    data_table['date'] = data_table['date'].str[0:DATE_LENGTH]

def split_featurs_and_lables(data):
    return (np.delete(data,[LABEL_COLUMN],1), data[:,LABEL_COLUMN])

data = read_data()
X, Y = split_featurs_and_lables(data)

x =1