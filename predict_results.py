# -*- coding: utf-8 -*-

import pandas as pd
import csv

from sklearn.pipeline import make_pipeline

from joblib import load

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

    ### Predict

    prediction_data_file = "./prediction_dataset_87descriptors.csv"
    result_file = "./predicted results.csv"
    output_columns = ['Active/Inactive 1', 'Active/Inactive 2']

    # Read feature columns
    feature_columns = []
    with open("feature columns.csv", newline='') as out:
         reader = csv.reader(out)
         # get the first line
         feature_columns = list(reader)[0]

    predictions = {}

    for idx in range(0, len(output_columns)):

        output_column = output_columns[idx]

        ### Load saved model
        model_filename = "model for output " + str(idx) + ".joblib"
        mlp_pipeline = load(model_filename)

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
        print("Results saved to file: ", result_file)

if __name__ == '__main__':
    main()
