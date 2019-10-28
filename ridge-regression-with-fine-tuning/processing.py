import csv
import numpy as np
from implementations import ridge_regression
from helper_functions import load_csv_data
from helper_functions import compute_error
from helper_functions import compute_mse
from helper_functions import compute_rmse


def load(train_file, test_file):
    """
    Load dataset from the given path and build numpy array to form training and test data.
    :param train_file: file name/ path for input training data
    :param test_file: file name/ path for input testing data
    :return: features, targets, and indexes for training and testing
    """
    print('\nLoad the raw training and test set data...')
    y_train, tx_train, ids_train = load_csv_data(train_file)
    y_test, tx_test, ids_test = load_csv_data(test_file)
    print('\n... finished.')
    return y_train, tx_train, ids_train, y_test, tx_test, ids_test


def get_header(file):
    """
    Get the header line from the given file
    :param file: file name/ path
    :return: dict object specifying the first header line from the file
    """
    read_file = open(file, 'r')
    reader = csv.DictReader(read_file)
    return reader.fieldnames


def analyze(tx):
    """
    Analyze data by replacing null value, -999, with the median of non-null value in the
    certain column. Also, handle outliers by placing original value with upper and lower bound
    (mean +- std from a feature distribution). Finally, record the columns that have zero
    variance, which would be removed.
    :param tx: raw training data
    :return: the list of columns which will be deleted
    """
    num_cols = tx.shape[1]
    print('\nNumber of columns in the data matrix: ',num_cols)
    columns_to_remove = []
    print('Analysis of data:\n')
    for col in range(num_cols):
        current_col = tx[:, col]
        if len(np.unique(current_col)) == 1:
            print('The column with index ', col, ' is all the same, it will be removed.')
            columns_to_remove.append(col)
        else:
            current_col[current_col == -999] = np.median(current_col[current_col != -999])
            # Handling the outliers
            std_current_col = np.std(current_col)
            mean_current_col = np.mean(current_col)
            lower_bound = mean_current_col - 2 * std_current_col
            upper_bound = mean_current_col + 2 * std_current_col
            current_col[current_col < lower_bound] = lower_bound
            current_col[current_col > upper_bound] = upper_bound
            print('Null values in the ', col, ' indexed column are replaced with the mean and outliers.')
    return columns_to_remove


def remove_columns(tx, header, columns_to_remove):
    """
    Remove the columns recorded in the variable, col_to_remove, from training data tx and
    header.
    :param tx: an array of training data
    :param header: header array
    :param columns_to_remove: the list indicating which column to be removed
    :return: modified training data, tx, and header
    """
    print("\nRemove columns...")
    num_removed = 0
    for col in columns_to_remove:
        tx = np.delete(tx, col - num_removed, 1)
        header = np.delete(header, col - num_removed + 2)
        num_removed += 1
    print("\n... finished.")
    return tx, header


def create_csv(output_file, y, tx, ids, header, is_test):
    """
    Split the given dataset such that only the data points with a certain
    jet number remains, note that jet number is a discrete valued feature. In
    other words, filter the dataset using the jet number.
    :param y: known label data
    :param tx: an array of training data
    :param ids: an array of index of data
    :param jet_num: discrete integer value for some feature
    :return: an numpy array of labels, training data, and index having specified the certain jet number
    """
    print('\nCreate new csv file named ' + str(output_file) + '...')
    with open(output_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter = ',', fieldnames = header)
        writer.writeheader()
        for idx, y_row, tx_row in zip(ids, y, tx):
            if is_test:
                prediction = '?'
            else:
                prediction = 'b' if y_row == -1 else 's'
            dictionary = {'Id': int(idx),'Prediction': prediction}
            for index in range(len(tx_row)):
                dictionary[header[index + 2]] = float(tx_row[index])
            writer.writerow(dictionary)
        print('\n... finished.')


def split_data(y, tx, ids, jet_num):
    """
    Split the given dataset such that only the data points with a certain
    jet number remains, note that jet number is a discrete valued feature. In
    other words, filter the dataset using the jet number.
    :param y: known label data
    :param tx: an array of training data
    :param ids: an array of index of data
    :param jet_num: discrete integer value for some feature
    :return: an numpy array of labels, training data, and index having specified the certain jet number
    """
    mask = tx[:, 22] == jet_num
    return y[mask], tx[mask], ids[mask]


def process_data(train_file, test_file):
    """
    Create 4 new training dataset files and 4 new test dataset files.
    First, split the initial data tests using the discrete valued feature jet number,
    which can only take the values 0, 1, 2 and 3. Second, process the split data
    sets by replacing null values and deleting zero variance features.
    :param train_file: file name/ path for input training data
    :param test_file: file name/ path for input testing data
    """
    y_train, tx_train, ids_train, y_test, tx_test, ids_test = load(train_file, test_file)
    header_train = get_header(train_file)
    header_test = get_header(test_file)
    print('\nData set will be split into four, each representing data with different jet numbers.')
    for jet_num in range(4):
        print('\nProcess training set with jet number = ' + str(jet_num) + '...')
        y_train_jet, tx_train_jet, ids_train_jet = split_data(y_train, tx_train, ids_train, jet_num)
        columns_to_remove = analyze(tx_train_jet)
        tx_train_jet, header_train_jet = remove_columns(tx_train_jet, header_train, columns_to_remove)
        create_csv('train_jet_' + str(jet_num) + '.csv', y_train_jet, tx_train_jet, ids_train_jet, header_train_jet, False)
        print('\n... created train_jet_' + str(jet_num) + '.csv file.')
        print('\nProcess test set with jet number = ' + str(jet_num) + '...')
        y_test_jet, tx_test_jet, ids_test_jet = split_data(y_test, tx_test, ids_test, jet_num)
        columns_to_remove = analyze(tx_test_jet)
        tx_test_jet, header_test_jet = remove_columns(tx_test_jet, header_test, columns_to_remove)
        create_csv('test_jet_' + str(jet_num) + '.csv', y_test_jet, tx_test_jet, ids_test_jet, header_test_jet, True)
        print('\n... created test_jet_' + str(jet_num) + '.csv file.')


def report_prediction_accuracy(y, tx, w_best, verbose=True):
    """
    Report the percentage of correct predictions of a model applied on a set of labels.
    :param y: labels
    :param tx: training data
    :param w: weights
    :return: accuracy of predictions on a dataset
    """
    predictions = tx.dot(w_best)
    predictions[predictions >= 0] = 1
    predictions[predictions < 0] = -1
    correct_percentage = np.sum(predictions == y) / float(len(predictions))
    if verbose:
        print('Percentage of correct predictions is: ', correct_percentage * 100, '%')
    return correct_percentage


def build_k_indices(y, k_fold, seed):
    """
    Randomly partitions the indices of the data set into k groups.
    :param y: labels
    :param k_fold: number of folds
    :param seed: random generator seed
    :return: an array of k sub-indices that are randomly partitioned
    """
    num_rows = y.shape[0]
    interval = int(num_rows / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_rows)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, augmented_tx, k_indices, k, lambda_, report_predictions=False):
    """
    Perform cross_validation for a specific test set from the partitioned set.
    :param y: label data
    :param augmented_tx: augmented features
    :param k_indices: An array of k sub-indices that are randomly partitioned
    :param k: number of folds
    :param lambda_: regularization parameters
    :param report_predictions: report prediction or not
    :return: root mean square of loss training error, prediction
    """
    y_test = y[k_indices[k]]
    y_train = np.delete(y, k_indices[k])
    augmented_tx_test = augmented_tx[k_indices[k]]
    augmented_tx_train = np.delete(augmented_tx, k_indices[k], axis = 0)
    w, loss_train = ridge_regression(y_train, augmented_tx_train, lambda_)
    pred = report_prediction_accuracy(y_test, augmented_tx_test, w, False)
    return compute_rmse(loss_train), pred


def report_prediction_accuracy_logistic(y, tx, w_best, verbose=True):
    """
    Report the percentage of correct predictions of a model that is applied
    on a set of labels. This method specifically works for logistic regression
    since the prediction assumes that labels are between 0 and 1.
    :param y: labels
    :param tx: training data
    :param w_best: Optimized weight vector of the model
    :return: the percentage of correct predictions of the model when it is applied on the given test set of labels
    """
    predictions = tx.dot(w_best)
    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0
    correct_percentage = np.sum(predictions == y) / float(len(predictions))
    if verbose:
        print('Percentage of correct predictions is: ',correct_percentage * 100, '%')
    return correct_percentage


def train_test_split(y, tx, ratio, seed=1):
    """
    Split a given training data set to a test set and a training set,
    the sizes of the created sets are determined by the given ration.
    :param y: labels
    :param tx: training data
    :param ratio: ratio for splitting training and testing data
    :param seed: random seed
    :return: numpy array of training and testing data
    """
    np.random.seed(seed)
    permutation = np.random.permutation(len(y))
    shuffled_tx = tx[permutation]
    shuffled_y = y[permutation]
    split_position = int(len(y) * ratio)
    tx_training, tx_test = shuffled_tx[: split_position], shuffled_tx[split_position:]
    y_training, y_test = shuffled_y[: split_position], shuffled_y[split_position:]
    return y_training, tx_training, y_test, tx_test


def standardize(x, mean_x=None, std_x=None):
    """
    Standardize original data from the dataset.
    :param x: data to standardize
    :param mean_x: mean value of data given by the dataset
    :param std_x: standard deviation of data given by the dataset
    :return: standardized data
    """
    if mean_x is None:
        mean_x = np.mean(x,axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x,axis=0)
    x = x / std_x
    return x, mean_x, std_x


def min_max_normalization(x, min_x = None, max_x = None):
    """
    Normalize original data using the minimum and maximum value in the dataset
    :param x: data to normalize
    :param min_x: minimum value of data
    :param max_x: maximum value of data
    :return: normalized data
    """
    if min_x is None:
        min_x = np.min(x, axis=0)
    if max_x is None:
        max_x = np.max(x, axis=0)
    return (x - (min_x)) / (max_x - min_x), min_x, max_x


def change_labels_logistic(y):
    """
    The labels in logistic regression are interpreted as probabilities,
    so this method transfers the labels to the range [0, 1]
    :param y: labels
    :return: labels as probability
    """
    y[y == -1] = 0
    return y
