# -*- coding: utf-8 -*-

from time import time
# import sys, os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
# from sklearn.exceptions import ConvergenceWarning
# from sklearn.utils._testing import ignore_warnings
# import warnings

from joblib import dump

def perform_grid_search(mlp_pipeline, x, y):
    '''
    Performs a grid search for hyper-parameters of the MLPRegressor
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

    Parameters
    ----------
    mlp_pipeline : sklearn.pipeline
        MLPRegressor pipelie.
    x : dataframe
        dataframe with features
    y : dataframe
        dataframe with results

    Returns
    -------
    search : GridSearchCV
        resulting grid search

    '''
    # to see all nested hyper-parameters for the pipeline: mlp_pipeline.get_params().keys()
    print("Grid search for hyper-parameters...")
    tic = time()

    alphas = 10.0 ** -np.arange(0, 3) #np.logspace(-4, 0, 5)
    param_grid = {
        'mlpclassifier__alpha': alphas,
        'mlpclassifier__hidden_layer_sizes': [(50, 25, 5), (40, 20), (20, 5), (20,), (10,)]
    }

    n_folds = 3
    search = GridSearchCV(mlp_pipeline, param_grid, cv=n_folds,
                          n_jobs=-1, refit=True, return_train_score=True,
                          verbose=3)
    search.fit(x, y)

    ### print scores
    train_scores = search.cv_results_['mean_train_score']
    scores = search.cv_results_['mean_test_score']
    scores_std = search.cv_results_['std_test_score']
    print("Grid scores:\n")
    for score, std, train_score, params in zip(scores, scores_std, train_scores, search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) [train score: %0.3f] for %r"  % (score, std * 2, train_score, params))
    print("Best parameters: ", search.best_params_)
    print()
    print("Grid search is done in {:.3f}s".format(time() - tic))
    print("Refitting is done in   {:.3f}s".format(search.refit_time_))
    print("Best R2 test score:    {:.3f}".format(search.best_score_))
    ### plot CV score
    # plt.figure().set_size_inches(8, 6)
    # plt.semilogx(alphas, search.cv_results_['mean_train_score'], 'r-')
    # plt.semilogx(alphas, scores)
    # # plot error lines showing +/- std. errors of the scores
    # std_error = scores_std / np.sqrt(n_folds)
    # plt.semilogx(alphas, scores + std_error, 'b--')
    # plt.semilogx(alphas, scores - std_error, 'b--')
    # # alpha=0.2 controls the translucency of the fill color
    # plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)
    # plt.ylabel('CV score +/- std error')
    # plt.xlabel('alpha')
    # plt.axhline(1.0, linestyle='--', color='.5')
    # plt.axhline(np.max(scores), linestyle='--', color='.5')
    # plt.xlim([alphas[0], alphas[-1]])
    # plt.show()
    return search

def train_model(training_data_file, output_columns, hyperparameter_tuning):
    '''
    Train the model using the gread search for hyper-parameter tuning

    Parameters
    ----------
    training_data_file : string
        a csv file with training and test data

    hyperparameter_tuning : boolean
        whether to perform the hyperparameter tuning or use the model with the specified parameters obtained from the previous hyperparameter tuning

    Returns
    -------
    mlp_pipeline : sklearn pipeline
        a pipeline model.
    feature_columns : list
        a list of features

    '''
    ### Load data from csv file
    # load the .csv file into a dataframe
    df = pd.read_csv(training_data_file)
    feature_columns = list(df.columns[5:])

    # find Nan values and exclude such rows
    if df.isnull().values.any():
        print('The data have Nan values')

    X = pd.DataFrame(df, columns = feature_columns)
    y = pd.DataFrame(df, columns = output_columns)

    ### Divide the dataset into the training and validation subsets
    # 70-80% for training and 30-20% for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    #y_1d = [int(i) for i in y[output_columns[0]]]
    y_train_1d = [int(i) for i in y_train[output_columns[0]]]
    y_test_1d = [int(i) for i in y_test[output_columns[0]]]

    ### Contruct NN
    # https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    # input neurons -> hiddent layers with neurons -> output neurons
    print("Training...")

    # Normalize the data for numerical stability - scale the data.
    # StandardScaler() - Standardize features by removing the mean and scaling to unit variance
    # Help on StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    # Note that we normalize in the pipeline, after splitting the data. It's good practice to apply any data transformations to training and testing data separately to prevent data leakage.
    scaler = StandardScaler()

    # Create Multi-layer Perceptron (MLP) Regressor
    # Help on MLPRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    #model = MLPRegressor(
    #        hidden_layer_sizes=(4,),        # number of neurons per hidden layer
    #        activation=activation_func,     # activation function
    #        solver='adam',                  # 'lbfgs' doesn't work well
    #        learning_rate='adaptive',
    #        learning_rate_init=1e-2,
    #        alpha=1e-0,                     # L2 regularization - penalty for weights with large magnitudes
    #        max_iter=int(1e6),
    #        # early_stopping=True,
    #        verbose=False)
    model = MLPClassifier(solver = 'adam',
                          max_iter = 1000,
                          learning_rate = 'adaptive',
                          alpha = 1e-0,
                          hidden_layer_sizes = (20, ),
                          random_state = 1,
                          verbose = False)

    # Create pipeline
    mlp_pipeline = make_pipeline(scaler, model)

    if hyperparameter_tuning:
        ### Grid search for hyper-parameters
        # it will automatically split data into training and test sets
        search = perform_grid_search(mlp_pipeline, X_train, y_train_1d)
        # Get the best estimator
        mlp_pipeline = search.best_estimator_
        model = mlp_pipeline['mlpclassifier']
    else:
        ### Train the model
        tic = time()
        mlp_pipeline.fit(X_train, y_train_1d)
        print("Training is done in {:.3f}s".format(time() - tic))
        print("Training R2 score:  {:.3f}".format(mlp_pipeline.score(X_train, y_train_1d)))
        print("Test R2 score:      {:.3f}".format(mlp_pipeline.score(X_test, y_test_1d)))

    ### Info on the NN
    # shape of NN
    print("Shape of the NN's coeficients:", [coef.shape for coef in model.coefs_])

    # Validate

    scores = {}
    print('Validating...')
    y_predicted = mlp_pipeline.predict(X_test)
    y_predicted = [int(i) for i in y_predicted]
    # Calculate the accuracy score by comparing the actual values and predicted values.
    # Calculate how well the model performed by comparing the model's predictions to the true target values, which we reserved in the y_test variable.
    TN, FP, FN, TP = confusion_matrix(y_true = y_test, y_pred = y_predicted).ravel()
    # Calculate metrics
    scores['accuracy'] = accuracy_score(y_true = y_test, y_pred = y_predicted)     # (TP + TN) / (TP + FP + TN + FN)
    scores['precision'] = precision_score(y_true = y_test, y_pred = y_predicted)   # TP / (TP + FP)
    scores['recall'] = recall_score(y_true = y_test, y_pred = y_predicted)         # TP / (TP + FN)

    print('\tTrue Positive(TP)  = ', TP)
    print('\tFalse Positive(FP) = ', FP)
    print('\tTrue Negative(TN)  = ', TN)
    print('\tFalse Negative(FN) = ', FN)
    print('\tAccuracy           = {:0.3f}'.format(scores['accuracy']))
    print('\tPrecision          = {:0.3f}'.format(scores['precision']))
    print('\tRecall             = {:0.3f}'.format(scores['recall']))

    return mlp_pipeline, feature_columns, output_columns, scores

def predict_using_trained_model(model, prediction_data_file, feature_columns, output_columns):
    '''
    Predict output for the given input data using the given

    Parameters
    ----------
    model : Regression model
        a trained model
    feature_columns : list
        a list of features
    prediction_data_file : csv file
        a csv file with input data

    Returns
    -------
    df_predict : dataframe
        a dataframe with predicted results

    y_predict : list
        a list of predicted values

    '''
    ### Predict
    print("\nPredicting...")
    df_predict = pd.read_csv(prediction_data_file)
    y_predict = []
    if df_predict.isnull().values.any():
        print('The data for predictions have NaN values. Please check the data for predictions.')
    else:
        X_predict = pd.DataFrame(df_predict, columns = feature_columns)
        y_predict = model.predict(X_predict)
        print(y_predict)
        for idx, value in enumerate(output_columns):
            df_predict[value] = y_predict
    return df_predict, y_predict

def main():

    ### Train model
    training_data_file = "./ML_set_119_reduced91_targets_corrected_CLASSES_0_1_87descriptors.csv"

    ### Predict
    prediction_data_file = "./prediction_dataset_87descriptors.csv"
    result_file = "./predicted results.csv"

    # output columns
    output_columns = ['Active/Inactive 1', 'Active/Inactive 2']

    # perform hyper-parameter tuning to find optimal parameters
    hyperparameter_tuning = False

    accuracy, precision, recall = {}, {}, {}
    predictions = {}

    for idx in range(0, len(output_columns)):

        output_column = output_columns[idx]

        mlp_pipeline, feature_columns, model_output_columns, scores = train_model(training_data_file, [output_column], hyperparameter_tuning)

        accuracy[output_column] = scores['accuracy']
        precision[output_column] = scores['precision']
        recall[output_column] = scores['recall']

        with open("feature columns.csv", 'w', newline='') as out:
             wr = csv.writer(out, quoting = csv.QUOTE_ALL)
             wr.writerow(feature_columns)

        model_filename = "model for output " + str(idx) + ".joblib"
        ### Save the model
        dump(mlp_pipeline, model_filename)

        df_predicted, y_predicted = predict_using_trained_model(mlp_pipeline, prediction_data_file, feature_columns, [output_column])

        y_predicted = [int(i) for i in y_predicted]
        predictions[output_column] = y_predicted

    # save results in a file
    print('Saving predicted results...')
    df_predict = pd.read_csv(prediction_data_file)
    if df_predict.isnull().values.any():
        print('The data for predictions have NaN values. Please check the data for predictions.')
    else:
        counter = 0
        for key in predictions:
            df_predict.insert(2 + counter, key, predictions[key])
            counter += 1
        df_predict.to_csv(result_file, sep = ",", index = False)

    # save scores in a csv file
    df_scores = pd.DataFrame(index = output_columns, columns = ['Accuracy', 'Precision', 'Recall'])
    df_scores['Accuracy'] = accuracy.values()
    df_scores['Precision'] = precision.values()
    df_scores['Recall'] = recall.values()
    df_scores.to_csv('scores.csv')

    # plot the scores
    ax = df_scores.plot.barh()
    ax.legend(
        ncol = len(output_columns),
        bbox_to_anchor = (0, 1),
        loc = 'lower left',
        prop = {'size': 14}
    )
    plt.gcf().set_dpi(300)
    plt.tight_layout()
    plt.savefig('scores.png')

if __name__ == '__main__':
    main()
