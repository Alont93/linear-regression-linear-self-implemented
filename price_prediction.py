import numpy as np
from numpy import random as rd
import pandas as pd
import matplotlib.pyplot as plt

DATE_LENGTH = 8
LABEL_COLUMN = 2
DATA_PATH = './kc_house_data.csv'
NORMALIZE_SAMPLES = False


def np_read_data():
    return np.genfromtxt(DATA_PATH, delimiter=',',skip_header=1)


def read_data():
    data_table = pd.read_csv(DATA_PATH)
    fix_dates(data_table)
    data_table = handle_categorical_variables(data_table)
    data_mat = data_table.values
    return data_mat.astype(np.float64)


def handle_categorical_variables(data_table):
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
        columns_ptp = np.ptp(X, axis=0)
        columns_ptp[columns_ptp == 0] = 1
        X = (X - np.min(X, axis=0)) / columns_ptp

    return np.linalg.pinv(X) @ y


def calculate_loss(prediction, labels):
    return (np.linalg.norm(prediction - labels, ord=2) ** 2) / labels.size


def predict_prices(data, hypothesis):
    return data @ hypothesis


def randomly_split_data(data, percent_of_training):
    data_size = data.shape[0]
    amount_of_training_data = int(data_size * percent_of_training * 0.01)
    training_set_indices = rd.choice(data_size, size=amount_of_training_data, replace=False)

    training_data = data[training_set_indices, :]
    test_data = np.delete(data, training_set_indices, axis=0)

    return training_data, test_data


def q8():
    data = read_data()
    results = []

    for i in range(1, 100):
        training_data, test_data = randomly_split_data(data, i)
        X_train, Y_train = split_samples_and_lables(training_data)
        X_test, Y_test = split_samples_and_lables(test_data)

        X_train = add_affine_parameter_to_samples(X_train)
        X_test = add_affine_parameter_to_samples(X_test)

        hypothesis = find_approximation_vector(X_train, Y_train)
        predicted_prices = predict_prices(X_test, hypothesis)
        loss = calculate_loss(predicted_prices, Y_test)

        results.append(loss)
        print(i)

    print(results)
    plt.plot(results)
    plt.xlabel("percentage of training data")
    plt.ylabel("loss")
    plt.title("Loss as a function of the percentage of training data")
    plt.show()

q8()

# data = read_data()
# X, Y = split_samples_and_lables(data)
# X = add_affine_parameter_to_samples(X)
# w = find_approximation_vector(X, Y)

x = 1