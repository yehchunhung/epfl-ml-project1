import numpy as np
import argparse
from processing import process_data
from processing import report_prediction_accuracy
from processing import build_k_indices
from processing import cross_validation
from processing import standardize
from helper_functions import build_polynomial
from helper_functions import feature_engineering
from helper_functions import load_csv_data
from helper_functions import predict_labels
from helper_functions import create_csv_submission
from implementations import *

# File names of the processed and split data sets
training_files = ['train_jet_0.csv', 'train_jet_1.csv', 'train_jet_2.csv', 'train_jet_3.csv']
test_files = ['test_jet_0.csv', 'test_jet_1.csv', 'test_jet_2.csv', 'test_jet_3.csv']


# main method
def main(am, t):
    if am:
        process_data('train.csv', 'test.csv')
    y_train_jets = []
    tx_train_jets = []
    ids_train_jets = []
    y_test_jets = []
    tx_test_jets = []
    ids_test_jets = []
    load_data_sets(y_train_jets, tx_train_jets, ids_train_jets, y_test_jets, tx_test_jets, ids_test_jets)
    degree_best_jets = [6, 6, 6, 6]
    lambda_best_jets = [3e-05, 0.0023, 4.6e-9, 5.7e-05]
    if t:
        grid_search_cross_validation(degree_best_jets, lambda_best_jets, y_train_jets, tx_train_jets)
    predictions = []
    ids_predicted = []
    learn(predictions, ids_predicted, y_train_jets, tx_train_jets, tx_test_jets, ids_test_jets, lambda_best_jets, degree_best_jets)
    combine_and_create_submission(predictions, ids_predicted, 'submit_TWN1')


# load each data set for each jet number
def load_data_sets(y_train_jets, tx_train_jets, ids_train_jets, y_test_jets, tx_test_jets, ids_test_jets):
    print('\nLoad the training and test set data of each jet number...')
    for jet_num in range(4):
        y_train, tx_train, ids_train = load_csv_data(training_files[jet_num])
        y_train_jets.append(y_train)
        tx_train_jets.append(tx_train)
        ids_train_jets.append(ids_train)
        y_test, tx_test, ids_test = load_csv_data(test_files[jet_num])
        y_test_jets.append(y_test)
        tx_test_jets.append(tx_test)
        ids_test_jets.append(ids_test)
        print('\nTraining and test set data for jet ', str(jet_num), ' is loaded.')
    print('\n... done.')


# perform a comprehensive grid search using cross validation to tune the hyperparameters
def grid_search_cross_validation(degree_best_jets, lambda_best_jets, y_train_jets, tx_train_jets):
    k_fold = 10
    degrees = np.arange(6, 10)
    lambda_powers = np.arange(-10, -1)
    lambda_numbers = np.arange(10, 100)
    for jet_num in range(4):
        max_pred = 0
        lambda_best = None
        degree_best = None
        print('\nGrid search with cross validation for jet ', str(jet_num), '...')
        y_train, tx_train = y_train_jets[jet_num], tx_train_jets[jet_num]
        k_indices = build_k_indices(y_train, k_fold, seed=1)
        for degree in degrees:
            print('\nSwitch to new degree: ', degree)
            expanded_tx_train = build_polynomial(tx_train, degree_best_jets[jet_num])
            for lambda_power in lambda_powers:
                print('\nSwitch to new lambda power: ', lambda_power)
                for lambda_number in lambda_numbers:
                    lambda_ = (lambda_number / 10.) * (10. ** lambda_power)
                    test_preds = []
                    for k in range(k_fold):
                        _, test_pred = cross_validation(y_train, expanded_tx_train, k_indices, k, lambda_)
                        test_preds.append(test_pred)
                    mean_test_pred = np.mean(test_preds)
                    if mean_test_pred > max_pred:
                        print('\nBetter hyperparameters found:')
                        max_pred = mean_test_pred
                        degree_best = degree
                        lambda_best = lambda_
                        print('- polynomial degree = ', degree_best)
                        print('- lambda = ', lambda_best)
                        print('\n- new best prediction percentage = ', max_pred * 100, '%')
        print('\nOptimal hyperparameters:')
        print('- polynomial degree = ', degree_best)
        degree_best_jets[jet_num] = degree_best
        print('- lambda = ', lambda_best)
        lambda_best_jets[jet_num] = lambda_best
        print('\n- best correct prediction percentage = ', max_pred * 100, '%')
        print('\n... finished')


# construct the model and the prediction using the optimal hyperparameter with ridge regression
def learn(predictions, ids_predicted, y_train_jets, tx_train_jets, tx_test_jets, ids_test_jets,
          lambda_best_jets, degree_best_jets):
    print('\nStart Learning')
    for jet_num in range(4):
        print('\nLearning from training set with jet number ', str(jet_num),'...')
        y_train, tx_train = y_train_jets[jet_num], tx_train_jets[jet_num]
        tx_train = feature_engineering(tx_train, degree_best_jets[jet_num], jet_num > 1)
        w_best, _ = ridge_regression(y_train, tx_train, lambda_best_jets[jet_num])
        tx_test, ids_test = tx_test_jets[jet_num], ids_test_jets[jet_num]
        tx_test = feature_engineering(tx_test, degree_best_jets[jet_num], jet_num > 1)
        predictions.append(predict_labels(w_best, tx_test))
        ids_predicted.append(ids_test)
        print('\nPrediction accuracy for the training set... \n')
        report_prediction_accuracy(y_train, tx_train, w_best)
        print('\n... this gives an estimation on the training.')
        print('\n... predicted labels for test set with jet number ', str(jet_num))
    print('\n... ,predicted labels for each test set.')


# combine the predictions of individual jets together and creates a submission file
def combine_and_create_submission(predictions, ids_predicted, submission_name):
    ids_gathered = []
    predictions_gathered = []
    current_id = min(ids_predicted[:][0])
    length = np.sum(len(prediction) for prediction in predictions)
    print('\nCollect IDs and predictions for each jet number...')
    for _ in range(length):
        for jet_num in range(4):
            if len(ids_predicted[jet_num]) > 0:
                if ids_predicted[jet_num][0] == current_id:
                    predictions_gathered.append(predictions[jet_num][0])
                    ids_gathered.append(current_id)
                    predictions[jet_num] = np.delete(predictions[jet_num], 0)
                    ids_predicted[jet_num] = np.delete(ids_predicted[jet_num], 0)
                    break
        current_id += 1
    print('\n... IDs and predictions for each jet number are collected.')
    print('\n Create submission file with name ', str(submission_name), ' ...')
    create_csv_submission(np.array(ids_gathered), np.array(predictions_gathered), submission_name)
    print('\n... ', str(submission_name), ' is created. Ready to submit!')


# command line arguments are defined here
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs the procedure to obtain group TaiwanNo1\'s best submission')
    parser.add_argument('-am', action='store_true', help='Split, analyze and modify raw data',
                        default=False)
    parser.add_argument('-t', action='store_true',
                        help='Tune the hyperparameter using grid search with cross validation. (Hard coded' +
                             ' optimal hyperparamete will be used if this argument is omitted)',
                        default=False)
    args = parser.parse_args()
    main(args.am,args.t)
