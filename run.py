# -*- coding: utf-8 -*-

"""
ML project 1 - Spot the Boson
This will produce exactly the same .csv predictions
which we used in our best submission to the competition system.
"""

import numpy as np
from nn import *
from proj1_helpers import *


def main():
    """ Main function """
    # read training and testing data
    DATA_TRAIN_PATH = '../../data/train.csv'
    DATA_TEST_PATH = '../../data/test.csv'
    y, x, _ = load_csv_data(DATA_TRAIN_PATH)
    _, x_test, ids_test = load_csv_data(DATA_TEST_PATH)

    # split training data into training and validation
    y_train, y_val, x_train, x_val = train_val_split(y, x, 0.2, seed=1)

    # normalize data using metrics of training data
    # (except PRI_jet_num (22th column) since it is a discrete value)
    nor_indices = [idx for idx in range(x_train.shape[1]) if idx != 22]
    nor_x_train = x_train.copy()
    nor_x_val = x_val.copy()
    nor_x_test = x_test.copy()

    x_train_mean = x_train[:, nor_indices].mean(axis=0)
    x_train_std = x_train[:, nor_indices].std(axis=0)

    nor_x_train[:, nor_indices] = (nor_x_train[:, nor_indices] - x_train_mean) / x_train_std
    nor_x_val[:, nor_indices] = (nor_x_val[:, nor_indices] - x_train_mean) / x_train_std
    nor_x_test[:, nor_indices] = (nor_x_test[:, nor_indices] - x_train_mean) / x_train_std

    # change y label from (1, -1) to (1, 0)
    # since the output of sigmoid is between (1, 0)
    nn_y_train = np.where(y_train == -1, 0, y_train)
    nn_y_val = np.where(y_val == -1, 0, y_val)

    # set parameters
    epochs = 100
    lr = 0.001
    batch_size = 250

    # train model, get model parameters
    nn_params = train(nor_x_train.T, nn_y_train[np.newaxis,:], nor_x_val.T, nn_y_val[np.newaxis,:],
                    NN_ARCHITECTURE, epochs, lr, batch_size, verbose=True)

    # read the best model in terms of accuracy and get the output of it
    nn_params = np.load('best_acc.npy', allow_pickle=True).item()
    y_test_hat, _ = full_forward_propagation(nor_x_test.T, nn_params, NN_ARCHITECTURE)
    # get the prediction labels
    y_test_pred = np.where(np.squeeze(y_test_hat.T) > 0.5, 1, -1)

    # save our prediction
    OUTPUT_PATH = 'test.csv'
    create_csv_submission(ids_test, y_test_pred, OUTPUT_PATH)


if __name__ == '__main__':

    main()
