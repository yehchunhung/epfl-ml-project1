# -*- coding: utf-8 -*-

"""Implementation of ML methods."""


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    raise NotImplementedError


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent."""
    # Define parameters to store w and loss
    raise NotImplementedError


def least_squares(y, tx):
    """Least squares regression using normal eqations."""
    raise NotImplementedError


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    raise NotImplementedError


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD."""
    raise NotImplementedError


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD."""
    raise NotImplementedError
