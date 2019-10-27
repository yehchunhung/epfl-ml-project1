# -*- coding: utf-8 -*-

"""Some helper functions for project 1."""

import csv
import numpy as np
import pandas as pd
from activations import sigmoid


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def predict_lg_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = sigmoid(np.dot(data, weights))
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def train_val_split(y, tx, val_per, seed=1):
    """Split data set into training set and validation set"""
    np.random.seed(seed)
    total_num = len(y)
    val_num = int(total_num * val_per)
    # get a random sequence of indices
    indices = np.random.permutation(total_num)
    val_idx, train_idx = indices[:val_num], indices[val_num:]
    return y[train_idx], y[val_idx], tx[train_idx, :], tx[val_idx, :]


def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def compute_accuracy(y_true, y_pred):
    """Compute accuracy"""
    return sum(y_true == y_pred) / len(y_true)


def compute_ls_loss(y, tx, w):
    """Compute mean squared error"""
    e = y - tx.dot(w)
    loss = 1 / 2 * np.mean(e**2)
    return loss


def compute_ls_gradient(y, tx, w):
    """Compute the gradient and loss."""
    e = y - tx.dot(w)
    grad = -1 / len(e) * tx.T.dot(e)
    return grad


def compute_lg_loss(y, tx, w):
    """Compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(-loss) / y.shape[0]


def compute_lg_gradient(y, tx, w):
    """Compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad / y.shape[0]


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def nn_batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Because the input shape is different from other models
    I need a new batch_iter for the neural network
    """
    data_size = y.shape[1]

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[:, shuffle_indices]
        shuffled_tx = tx[:, shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[:, start_index:end_index], shuffled_tx[:, start_index:end_index]

def standardize(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

def load_train_data_split(DATA_TRAIN_PATH):
    
    train = pd.read_csv(DATA_TRAIN_PATH)
    train['Prednum'] = np.where(train['Prediction'] == 'b', 0, 1)
    
    x_0 = train[train['PRI_jet_num'] == 0].drop(columns=['Id', 'Prediction','PRI_jet_num',
                                                             'DER_deltaeta_jet_jet','DER_mass_jet_jet',
                                                             'DER_prodeta_jet_jet', 'DER_lep_eta_centrality',
                                                             'PRI_jet_leading_pt','PRI_jet_leading_eta',
                                                             'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
                                                             'PRI_jet_subleading_eta','PRI_jet_subleading_phi',
                                                             'PRI_jet_all_pt','Prednum'])
    
    x_1 = train[train['PRI_jet_num'] == 1].drop(columns=['Id', 'Prediction', 'PRI_jet_num',
                                                             'DER_deltaeta_jet_jet','DER_mass_jet_jet',
                                                             'DER_prodeta_jet_jet','DER_lep_eta_centrality',
                                                             'PRI_jet_subleading_pt','PRI_jet_subleading_eta',
                                                             'PRI_jet_subleading_phi','PRI_jet_all_pt','Prednum'])
    
    x_2 = train[(train['PRI_jet_num'] == 2) | (train['PRI_jet_num'] == 3)].drop(columns=['Id', 'Prediction', 'PRI_jet_num','Prednum'])
     
    y = train[['Prednum', 'PRI_jet_num', 'DER_mass_MMC']]
    y_0 = y[y['PRI_jet_num'] == 0]['Prednum'].values
    y_1 = y[y['PRI_jet_num'] == 1]['Prednum'].values
    y_2 = y[(y['PRI_jet_num'] == 2) | (y['PRI_jet_num'] == 3)]['Prednum'].values
     
    x_0 = standardize(x_0)
    x_1 = standardize(x_1)
    x_2 = standardize(x_2)
    return x_0, x_1, x_2, y_0, y_1, y_2

def load_test_data_split(DATA_TEST_PATH):
    test = pd.read_csv(DATA_TEST_PATH)
    id_0 = test[test['PRI_jet_num'] == 0]["Id"]
    id_1 = test[test['PRI_jet_num'] == 1]["Id"]
    id_2 = test[(test['PRI_jet_num'] == 2) | (test['PRI_jet_num'] == 3)]["Id"]
    
    x_0 = test[test['PRI_jet_num'] == 0].drop(columns=['Id', 'Prediction', 'PRI_jet_num',
                                                             'DER_deltaeta_jet_jet','DER_mass_jet_jet',
                                                             'DER_prodeta_jet_jet', 'DER_lep_eta_centrality',
                                                             'PRI_jet_leading_pt','PRI_jet_leading_eta',
                                                             'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
                                                             'PRI_jet_subleading_eta','PRI_jet_subleading_phi',
                                                             'PRI_jet_all_pt'])
    x_1 = test[test['PRI_jet_num'] == 1].drop(columns=['Id', 'Prediction', 'PRI_jet_num',
                                                             'DER_deltaeta_jet_jet','DER_mass_jet_jet',
                                                             'DER_prodeta_jet_jet','DER_lep_eta_centrality',
                                                             'PRI_jet_subleading_pt','PRI_jet_subleading_eta',
                                                             'PRI_jet_subleading_phi','PRI_jet_all_pt'])
    
    x_2 = test[(test['PRI_jet_num'] == 2) | (test['PRI_jet_num'] == 3)].drop(columns=['Id','Prediction', 'PRI_jet_num'])
    
    x_0 = standardize(x_0)
    x_1 = standardize(x_1)
    x_2 = standardize(x_2)
    return x_0, x_1, x_2, id_0, id_1, id_2
