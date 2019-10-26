# -*- coding: utf-8 -*-

"""Activation functions."""

import numpy as np


def sigmoid(x):
    """Sigmoid implementation"""
    return 1 / (1 + np.exp(-x))


def relu(x):
    """RELU implementation"""
    return np.maximum(0, x)


def sigmoid_backward(dA, Z):
    """Sigmoid backward (derivative) implementation"""
    dsig = sigmoid(Z) *  (1 - sigmoid(Z))
    return dA * dsig


def relu_backward(dA, Z):
    """RELU backward (derivative) implementation"""
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ
