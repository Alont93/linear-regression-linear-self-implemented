import numpy as np
import pandas as pd

DATE_LENGTH = 8
LABEL_COLUMN = 2
DATA_PATH = './kc_house_data.csv'
NORMALIZE_SAMPLES = True


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
    # remove linear independent variable
    data_table.drop(data_table.columns[[-1,]], axis=1, inplace=True)
    return data_table


def fix_dates(data_table):
    data_table['date'] = data_table['date'].str[0:DATE_LENGTH]


def split_samples_and_lables(data):
    return (np.delete(data,[LABEL_COLUMN],1), data[:,LABEL_COLUMN])


def add_affine_parameter_to_samples(samples):
    return np.hstack((samples,np.ones((samples.shape[0],1))))


def find_approximation_vector(samples, labels):
    X = samples
    y = labels

    if NORMALIZE_SAMPLES:
        X[:,0:-1] = (X[:,0:-1] - np.min(X[:,0:-1] ,axis=0)) / np.ptp(X[:,0:-1], axis=0)

    return np.linalg.pinv(X) @ y


data = read_data()
X, Y = split_samples_and_lables(data)
X = add_affine_parameter_to_samples(X)
w = find_approximation_vector(X, Y)


x = 1