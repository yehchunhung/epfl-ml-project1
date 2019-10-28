# -*- coding: utf-8 -*-

"""Implementation of ML methods."""

import numpy as np
from proj1_helpers import compute_ls_loss, compute_ls_gradient, batch_iter
from proj1_helpers import compute_lg_loss, compute_lg_gradient


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    w = initial_w
    losses = []
    threshold = 1e-8
    for n_iter in range(max_iters):
        # compute loss and gradient
        loss = compute_ls_loss(y, tx, w)
        grad = compute_ls_gradient(y, tx, w)
        # update w by gradient
        w = w - gamma * grad
        # log info
        # print("Gradient Descent({bi}/{ti}): loss={l}".format(
        #     bi=n_iter, ti=max_iters - 1, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return (w, loss)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent."""
    # Define parameters to store w and loss
    w = initial_w
    losses = []
    threshold = 1e-8
    for n_iter in range(max_iters):
        # get a random minibatch of data
        for minibatch_y, minibatch_x in batch_iter(y, tx, 1):
            # compute loss and gradient
            loss = compute_ls_loss(minibatch_y, minibatch_x, w)
            grad = compute_ls_gradient(minibatch_y, minibatch_x, w)
            # update w by gradient
            w = w - gamma * grad
        # log info
        # print("Stochastic Gradient Descent({bi}/{ti}): loss={l}".format(
        #     bi=n_iter, ti=max_iters - 1, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return (w, loss)


def least_squares(y, tx):
    """Least squares regression using normal eqations."""
    lt = np.dot(tx.T, tx)
    rt = np.dot(tx.T, y)
    # solve normal equation
    w = np.linalg.solve(lt, rt)
    # compute loss
    loss = compute_ls_loss(y, tx, w)

    return (w, loss)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    # add regularization term
    reg = 2 * len(tx) * lambda_ * np.identity(tx.shape[1])
    lt = np.dot(tx.T, tx) + reg
    rt = np.dot(tx.T, y)
    # solve normal equation
    w = np.linalg.solve(lt, rt)
    # compute loss
    loss = compute_ls_loss(y, tx, w)

    return (w, loss)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD."""
    w = initial_w
    losses = []
    threshold = 1e-8
    for n_iter in range(max_iters):
        # compute loss and gradient
        loss = compute_lg_loss(y, tx, w)
        grad = compute_lg_gradient(y, tx, w)
        # update w
        w = w - gamma * grad
        # log info
        # if n_iter % 100 == 0:
        #     print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return (w, loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD."""
    w = initial_w
    losses = []
    threshold = 1e-8
    for n_iter in range(max_iters):
        # compute loss and gradient (adding regularization term)
        loss = compute_lg_loss(y, tx, w) + lambda_ / 2 * np.linalg.norm(w)**2
        grad = compute_lg_gradient(y, tx, w) + lambda_ * w
        # update w
        w = w - gamma * grad
        # log info
        # if n_iter % 100 == 0:
        #     print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return (w, loss)
