import numpy as np
from helper_functions import compute_error
from helper_functions import compute_mse
from helper_functions import compute_gradient
from helper_functions import compute_logistic_gradient
from helper_functions import compute_logistic_loss
from helper_functions import regularized_logistic_regression


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent
    :param y: labels
    :param tx: training data
    :param initial_w: initial value of weights
    :param max_iters: maximum iterations used in gradient descent process
    :param gamma: learning rate
    :return: optimized loss value based on MSE, optimized weight vectors for the model
    """
    # Define parameters to store w and loss
    threshold = 1e-9
    ws = [initial_w]
    losses = []
    w = initial_w
    for _ in range(max_iters):
        loss = compute_mse(compute_error(y, tx, w))
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
        ws.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return ws[-1], losses[-1]


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent
    :param y: labels
    :param tx: training data
    :param initial_w: initial value of weights
    :param max_iters: maximum iterations used in gradient descent process
    :param gamma: learning rate
    :return: optimized loss value based on MSE, optimized weight vectors for the model
    """
    threshold = 1e-9
    ws = [initial_w]
    losses = []
    w = initial_w
    for _ in range(max_iters):
        random_index = np.random.randint(len(y))
        y_random = y[random_index]
        tx_random = tx[random_index]
        error_vector = compute_error(y_random, tx_random, w)
        loss = compute_mse(error_vector)
        gradient = compute_gradient(tx_random, error_vector)
        w = w - gamma * gradient
        ws.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return ws[-1], losses[-1]


def least_squares(y, tx):
    """
    Least squares regression using the normal equation
    :param y: labels
    :param tx: training data
    :return: optimized loss value based on MSE, optimized weight vectors for the model
    """
    coefficient_matrix = tx.T.dot(tx)
    constant = tx.T.dot(y)
    w = np.linalg.solve(coefficient_matrix, constant)
    loss = compute_mse(compute_error(y, tx, w))
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using the normal equation
    :param y: labels
    :param tx: training data
    :param lambda_: regularization parameter
    :return: optimized loss value based on MSE, optimized weight vectors for the model
    """
    coefficient_matrix = tx.T.dot(tx) + 2 * len(y) * lambda_ * np.identity(tx.shape[1])
    constant_vector = tx.T.dot(y)
    w = np.linalg.solve(coefficient_matrix, constant_vector)
    error_vector = compute_error(y, tx, w)
    loss = compute_mse(error_vector)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using stochastic gradient descent
    :param y: labels
    :param tx: training data
    :param initial_w: initial weight value
    :param max_iters: maximum number of iterations
    :param gamma: learning rate
    :return: optimized weights and losses
    """
    threshold = 1e-9
    ws = [initial_w]
    losses = []
    w = initial_w
    for iter in range(max_iters):
        loss = compute_logistic_loss(y, tx, w)
        grad = compute_logistic_gradient(y, tx, w)
        w = w - (gamma*grad)
        ws.append(w)
        losses.append(loss)
        if (len(losses) > 1) and (len(ws) > 1) and (np.abs(ws[-1] - ws[-2]) <= threshold):
            break
    return ws[-1], losses[-1]


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent
    :param y: labels
    :param tx: training data
    :param lambda_: regularized hyperparameter
    :param initial_w: initial weight value
    :param max_iters: maximum number of iterations
    :param gamma: learning rate
    :return: optimized weights and loss
    """
    threshold = 1e-9
    ws = [initial_w]
    losses = []
    w = initial_w
    for iter in range(max_iters):
        loss, grad = regularized_logistic_regression(y, tx, w, lambda_)
        w = w - (gamma * grad)
        ws.append(w)
        losses.append(loss)
        if (len(losses) > 1) and (len(ws) > 1) and (np.abs(ws[-1] - ws[-2]) <= threshold):
            break
    return ws[-1], losses[-1]
