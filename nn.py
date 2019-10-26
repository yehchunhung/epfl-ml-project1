# -*- coding: utf-8 -*-

"""
Neural Network Implementation
I followed this tutorial
https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
and made some modifications
"""

import timeit
import numpy as np

from activations import sigmoid, relu, sigmoid_backward, relu_backward
from proj1_helpers import nn_batch_iter


# Specify the structure of the neural network
NN_ARCHITECTURE = [
    {"input_dim": 30, "output_dim": 64, "activation": "relu"},
    {"input_dim": 64, "output_dim": 64, "activation": "relu"},
    {"input_dim": 64, "output_dim": 1, "activation": "sigmoid"},
]


def init_layers(nn_architecture, seed=10):
    """Initiate layers and corresponding paramters"""
    # random seed initiation
    np.random.seed(seed)
    # parameters storage initiation
    params_values = {}

    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # number network layers from 1
        layer_idx = idx + 1

        # extracting the number of units in layers
        layer_input_size = layer['input_dim']
        layer_output_size = layer['output_dim']

        # initiating the values of the W matrix and vector b for subsequent layers
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1

    return params_values


def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation):
    """Calculate the output of a single layer forward propagation"""
    # calculate the input value for the activation function
    z_curr = W_curr.dot(A_prev) + b_curr

    # select activation function
    if activation == "relu":
        activation_func = relu
    elif activation == "sigmoid":
        activation_func = sigmoid

    # return activation a and the intermediate z matrix
    return activation_func(z_curr), z_curr


def full_forward_propagation(x, params_values, nn_architecture):
    """Perform forward propagation for the full network"""
    # creating a temporary memory to store the information needed for a backpropagation
    memory = {}
    # X vector is the activation for layer 0 
    A_curr = x

    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # number network layers from 1
        layer_idx = idx + 1
        # transfer the activation from the previous iteration
        A_prev = A_curr

        # extract the activation function for the current layer
        activ_function_curr = layer["activation"]
        # extract w and b for the current layer
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        # calculate the activation for the current layer
        A_curr, Z_curr = single_layer_forward_propagation(
            A_prev, W_curr, b_curr, activ_function_curr)

        # savie values in the memory
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr

    # return prediction vector and memory containing intermediate values
    return A_curr, memory


def single_layer_backward_propagation(dA_curr, W_curr, Z_curr, A_prev, activation):
    """Calculate the output of a single layer backward propagation"""
    # number of examples
    num = A_prev.shape[1]

    # select activation function (backward version)
    if activation is "relu":
        backward_activation_func = relu_backward
    elif activation is "sigmoid":
        backward_activation_func = sigmoid_backward

    # calculate all the derivatives
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dA_prev = W_curr.T.dot(dZ_curr)
    dW_curr = dZ_curr.dot(A_prev.T) / num
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / num

    return dA_prev, dW_curr, db_curr


def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    """Perform backward propagation for the full network"""
    grads_values = {}

    # calculate first derivative
    # The 1e-07 here is making sure it won't face division by zero
    dA_prev = - ((Y / (Y_hat + 1e-07)) - ((1 - Y) / (1 - Y_hat + 1e-07)))

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        # number network layers from 1
        layer_idx_curr = layer_idx_prev + 1
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]

        dA_curr = dA_prev

        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]

        W_curr = params_values["W" + str(layer_idx_curr)]

        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, Z_curr, A_prev, activ_function_curr)

        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr

    return grads_values


def update(params_values, grads_values, nn_architecture, learning_rate, prev_sigma, rho=0.9):
    """Update the parameters using gradient descent with RMSprop optimizer"""
    # the current sigma
    sigma = {}
    for layer_idx, layer in enumerate(nn_architecture, 1):
        layer_input_size = layer['input_dim']
        layer_output_size = layer['output_dim']
        sigma["W" + str(layer_idx)] = np.zeros((layer_output_size, layer_input_size))
        sigma["b" + str(layer_idx)] = np.zeros((layer_output_size, 1))
    # iteration over network layers
    for layer_idx, _ in enumerate(nn_architecture, 1):
        # first sigma should just be the current gradient
        if prev_sigma["W" + str(layer_idx)].all() == 0:
            # The 1e-07 here is making sure it won't face division by zero
            sigma["W" + str(layer_idx)] = grads_values["dW" + str(layer_idx)] + 1e-07
        else:
            sigma["W" + str(layer_idx)] = np.sqrt(rho * prev_sigma["W" + str(layer_idx)]**2
                + (1 - rho) * grads_values["dW" + str(layer_idx)]**2) + 1e-07
        # do the same things for updating bias
        if prev_sigma["b" + str(layer_idx)].all() == 0:
            sigma["b" + str(layer_idx)] = grads_values["db" + str(layer_idx)] + 1e-07
        else:
            sigma["b" + str(layer_idx)] = np.sqrt(rho * prev_sigma["b" + str(layer_idx)]**2
                + (1 - rho) * grads_values["db" + str(layer_idx)]**2) + 1e-07

        params_values["W" + str(layer_idx)] -= learning_rate / sigma["W" +
            str(layer_idx)] * grads_values["dW" + str(layer_idx)]
        params_values["b" + str(layer_idx)] -= learning_rate / sigma["b" +
            str(layer_idx)] * grads_values["db" + str(layer_idx)]

    return params_values, sigma


def compute_nn_loss(y_hat, y):
    """Compute logistic regression loss"""
    # number of examples
    num = y_hat.shape[1]
    # calculation of the loss according to the formula
    # The 1e-07 here is making sure there is no log(0)
    loss = -1 / num * (y.dot(np.log(y_hat + 1e-07).T) + (1 - y).dot(np.log(1 - y_hat + 1e-07).T))
    return np.squeeze(loss)


def compute_nn_accuracy(y_hat, y):
    """Compare prediction and ground truth and compute accuracy"""
    y_pred = np.where(y_hat > 0.5, 1, 0)
    return (y_pred == y).mean()


def train(x, y, x_val, y_val, nn_architecture, epochs, learning_rate, batch_size, verbose=False):
    """Neural network training"""
    # initiate parameters
    params_values = init_layers(nn_architecture, 2)
    # sigma for RMSprop optimizer
    prev_sigma = {}
    for layer_idx, layer in enumerate(nn_architecture, 1):
        layer_input_size = layer['input_dim']
        layer_output_size = layer['output_dim']
        prev_sigma["W" + str(layer_idx)] = np.zeros((layer_output_size, layer_input_size))
        prev_sigma["b" + str(layer_idx)] = np.zeros((layer_output_size, 1))

    # initiate lists storing the history of metrics calculated during the learning process
    loss_history = []
    val_loss_history = []
    acc_history = []
    val_acc_history = []

    start_time = timeit.default_timer()
    for i in range(epochs):
        i += 1
        b_loss_history = []
        b_acc_history = []
        # get random minibatches of data
        for b_y, b_x in nn_batch_iter(y, x, batch_size, int(y.shape[1] / batch_size)):
            # calculate the output of the model
            b_y_hat, memory = full_forward_propagation(b_x, params_values, nn_architecture)

            # calculate loss and accuracy
            b_loss = compute_nn_loss(b_y_hat, b_y)
            b_acc = compute_nn_accuracy(b_y_hat, b_y)

            # save them in history
            b_loss_history.append(b_loss)
            b_acc_history.append(b_acc)

            # calculate gradients
            grads_values = full_backward_propagation(
                b_y_hat, b_y, memory, params_values, nn_architecture)
            # update model weights with gradient descent and RMSprop optimizer
            params_values, prev_sigma = update(
                params_values, grads_values, nn_architecture, learning_rate, prev_sigma)

        y_val_hat, _ = full_forward_propagation(x_val, params_values, nn_architecture)

        # average of all the mini batches
        loss = np.mean(b_loss_history)
        acc = np.mean(b_acc_history)

        # calculate loss and accuracy
        val_loss = compute_nn_loss(y_val_hat, y_val)
        val_acc = compute_nn_accuracy(y_val_hat, y_val)

        # save best model parameters
        if i > 1:
            # in terms of validation accuracy
            if val_acc > max(val_acc_history) and val_acc > 0.832:
                print("Saving best model (epoch {}, val_acc: {:.4f})".format(i, val_acc))
                np.save('best_acc.npy', params_values)
            # in terms of validation loss
            if val_loss < min(val_loss_history) and val_loss < 0.375:
                print("Saving best model (epoch {}, val_loss: {:.4f})".format(i, val_loss))
                np.save('best_loss.npy', params_values)

        # save loss and accuracy in history
        loss_history.append(loss)
        val_loss_history.append(val_loss)
        acc_history.append(acc)
        val_acc_history.append(val_acc)

        # print info every 50 epochs
        if i % 50 == 0 or i == 1:
            end_time = timeit.default_timer()
            if verbose:
                print("Epoch: {:05} - {:.1f}s - loss: {:.4f} - acc: {:.4f} - val_loss: {:.4f} - val_acc: {:.4f}".format(
                    i, end_time - start_time, loss, acc, val_loss, val_acc))
            start_time = timeit.default_timer()

    return params_values
