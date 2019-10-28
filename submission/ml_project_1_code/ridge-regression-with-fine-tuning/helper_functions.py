import numpy as np
import csv


def load_csv_data(data_path, sub_sample=False):
    """
    load data and returns y (class labels), tX (features) and ids (event ids)
    :param data_path: path where the data file locates
    :param sub_sample: boolean variable to determine to choose subsample or not
    :return: output labels (-1, 1), input data, and its corresponding index
    """
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
    """Generate class predictions given weights, and a test data matrix
    :param weights: weight value for features
    :param data: input data
    :return: output prediction produced by weights and input data
    """
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Create an output file in csv format for submission to the competition
    :param ids: event ids associated with each prediction
    :param y_pred: string name of .csv output file to be created
    :param name: string name of .csv output file to be created
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def compute_error(y, tx, w):
   """
   Computes the error vector that is defined as y - tx . w
   :param y: labels
   :param tx: training data
   :param w: weights
   :return: error vector from y, tx, and w
   """
   return y - tx.dot(w)


def compute_mse(error):
    """
    Computes the mean squared error for a given error vector.
    :param error: error vector computed for a specific dataset and model
    :return: numeric value of the mean squared error
    """
    return np.mean(error ** 2) / 2


def compute_rmse(loss_mse):
    """
    Computes the root mean squared error.
    :param loss_mse: numeric value of the mean squared error loss
    :return: numeric value of the root mean squared error loss
    """
    return np.sqrt(2 * loss_mse)


def compute_gradient(y, tx, w):
    """
    Compute the gradient.
    :param y: labels
    :param tx: training data
    :param w: given weights
    :return: gradient of a loss, and loss with respect to y and tx
    """
    error = compute_error(y, tx, w)
    return - tx.T.dot(error) / error.size


def build_polynomial(x, degree):
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree.
    :param x: training data
    :param degree: numerical value indicating the degree to extend the basis features
    :return: polynomial extended basis features
    """
    num_cols = x.shape[1] if len(x.shape) > 1 else 1
    augmented_x = np.ones((len(x),1))
    for col in range(num_cols):
        for degree in range(1,degree + 1):
            if num_cols > 1:
                augmented_x = np.c_[augmented_x,np.power(x[:,col],degree)]
            else:
                augmented_x = np.c_[augmented_x,np.power(x,degree)]
        if num_cols > 1 and col != num_cols - 1:
            augmented_x = np.c_[augmented_x,np.ones((len(x),1))]
    return augmented_x


def sigmoid(t):
    """
    Apply sigmoid function on t
    :param t: argument inserted in exponential term
    :return: value of sigmoid function
    """
    return 1.0 / (1 + np.exp(-t))


def compute_logistic_gradient(y, tx, w):
    """
    Compute the gradient of loss
    :param y: labels
    :param tx: training data
    :param w: weights
    :return: gradient of logistic function
    """
    return tx.T.dot(sigmoid(tx.dot(w)) - y)


def compute_logistic_loss(y, tx, w):
    """
    Compute the cost by negative log likelihood
    :param y: labels
    :param tx: training data
    :param w: weights
    :return: loss of logistic regression
    """
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1-y).T.dot(np.log(1-pred))
    return np.squeeze(-loss)


def regularized_logistic_regression(y, tx, w, lambda_):
    """
    Return the loss, gradient
    :param y: labels
    :param tx: training data
    :param w: weights
    :param lambda_: hyperparamter for regularization
    :return: loss value and gradient
    """
    loss = compute_logistic_loss(y, tx, w) + 0.5*lambda_*(np.linalg.norm(w)**2)
    #loss = compute_logistic_loss(y, tx, w) + 0.5*lambda_*(np.linalg.norm(w))
    grad = compute_logistic_gradient(y, tx, w) + lambda_*w
    return loss, grad


def cross_terms(x, x_initial):
    """
    Add the multiplication of different features as new features.
    :param x: given feature matrix
    :param x_initial: features whose multiplication cross terms will be added
    :return: features with cross terms
    """
    for col1 in range(x_initial.shape[1]):
        for col2 in np.arange(col1 + 1,x_initial.shape[1]):
            if col1 != col2:
                x = np.c_[x,x_initial[:,col1] * x_initial[:,col2]]
    return x


def log_terms(x, initial_x):
    """
    Add the logarithm of features as new features.
    :param x: given feature matrix
    :param initial_x: features whose multiplication cross terms will be added
    :return: features in logarithm
    """
    for col in range(initial_x.shape[1]):
        current_col = initial_x[:, col]
        current_col[current_col <= 0] = 1
        x = np.c_[x, np.log(current_col)]
    return x


def sqrt_terms(x, x_initial):
    """
    Add the square roots of features as new features.
    :param x: given feature matrix
    :param x_initial: features whose square roots will be added
    :return: feature matrix with square roots
    """
    for col in range(x_initial.shape[1]):
        current_col = np.abs(x_initial[:, col])
        x = np.c_[x, np.sqrt(current_col)]
    return x


def apply_trigonometry(x, x_initial):
    """
    Add the sin and cos of features as new features.
    :param x: given feature matrix
    :param x_initial: features whose sin and cos will be added
    :return: feature matrix with sine values
    """
    for col in range(x_initial.shape[1]):
        x = np.c_[x, np.sin(x_initial[:, col])]
        x = np.c_[x, np.cos(x_initial[:, col])]
    return x


def feature_engineering(x, degree, has_angles = False):
    """
    Build a polynomial with the given degree from the initial features,
    add the cross terms, logarithms and square roots of the initial features
    as new features. Also includes the sine of features as an option.
    :param x: features
    :param degree: degree of the polynomial basis
    :param has_angles: Boolean variable to determine including sin and cos of features
    :return: features after doing feature engineering
    """
    x_initial = x
    x = build_polynomial(x, degree)
    x = cross_terms(x, x_initial)
    x = log_terms(x, x_initial)
    x = sqrt_terms(x, x_initial)
    if has_angles:
        x = apply_trigonometry(x, x_initial)
    return x
