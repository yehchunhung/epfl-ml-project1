# -*- coding: utf-8 -*-

"""
Main File for ML project 1
Spot the Boson
This will produce the .csv predictions which we used
in our best submission to the competition system.
"""

import numpy as np
from nn import *
from proj1_helpers import *


def main():
    """ Main function """
    # read testing data
    DATA_TEST_PATH = './data/test.csv'
    _, x_test, ids_test = load_csv_data(DATA_TEST_PATH)
    train_data_metrics = np.load('train_data_metrics.npy', allow_pickle=True).item()

    # normalize testing data with training data mean and std
    x_train_mean = train_data_metrics['mean']
    x_train_std = train_data_metrics['std']
    nor_indices = [idx for idx in range(x_test.shape[1]) if idx != 22]
    nor_x_test = x_test.copy()
    nor_x_test[:, nor_indices] = (nor_x_test[:, nor_indices] - x_train_mean) / x_train_std

    # read the best model and get the output of it
    nn_params = np.load('best_acc.npy', allow_pickle=True).item()
    y_test_hat, _ = full_forward_propagation(nor_x_test.T, nn_params, NN_ARCHITECTURE)
    # get the prediction labels
    y_test_pred = np.where(np.squeeze(y_test_hat.T) > 0.5, 1, -1)

    # save our prediction
    OUTPUT_PATH = 'test.csv'
    create_csv_submission(ids_test, y_test_pred, OUTPUT_PATH)


if __name__ == '__main__':

    main()
