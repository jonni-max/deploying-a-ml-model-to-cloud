"""
Script to train and test the machine learning model on the census dataset.

"""

from sklearn.model_selection import train_test_split
from joblib import dump, load

import numpy as np
import pandas as pd

# Add the necessary imports for the starter code.
from .ml.data import process_data
from .ml.model import train_model, inference_encoded
from .ml.model import compute_model_metrics
from scipy.special.tests.test_dd import test_data

# TODO Move these to a separate folder
g_fn_model = 'data/rf_model.joblib'
g_fn_encoder = 'data/encoder.joblib'
g_fn_label_binarizer = 'data/label_binarizer.joblib'

g_cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def model_training(training_data):
    """
    Runs the training component of the ml pipeline.

    Inputs
    ------
    training_data : pandas.DataFrame
        Input data for training and testing

    Returns
    -------
    model : np.array
        Model trained with the input data
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        One hot encoder
    label_binarizer : sklearn.preprocessing._label.LabelBinarizer
        Label binarizer
    """

    X_train, y_train, encoder, label_binarizer = process_data(
        training_data, categorical_features=g_cat_features, label="salary",
        training=True)

    # Train and save a model.
    model = train_model(X_train, y_train)

    return model, encoder, label_binarizer


def model_testing_overall(
        model,
        encoder,
        label_binarizer,
        test_data):
    """
    Evaluate performance of the overall model.
    
    Input
    -----
    model : sklearn.ensemble.RandomForestClassifier
        Model to be evaluated
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        One hot encoder
    label_binarizer : sklearn.preprocessing._label.LabelBinarizer
        Label binarizer
    test_data : pandas.DataFrame
        Test data with column names
        
    Returns
    -------
    
    """
    
    # Process the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test_data, categorical_features=g_cat_features, label="salary",
        training=False, encoder=encoder, lb=label_binarizer
    )

    # Compute model predictions on the whole test data
    preds = inference_encoded(model, X_test)
    
    prec, rec, fbet = compute_model_metrics(y_test, preds)
    
    result_text = f"Model performance:\nprecision: {prec}, recall: {rec}, "\
        f"fbeta: {fbet}"

    # Print result to stdout
    print(result_text)
    
        

def model_testing_slices(
        model,
        encoder,
        label_binarizer,
        test_data,
        feature_no):
    """
    Evaluate model performance on slices of one feature of the test data.

    Input
    -----
    model : sklearn.ensemble.RandomForestClassifier
        Model to be evaluated
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        One hot encoder
    label_binarizer : sklearn.preprocessing._label.LabelBinarizer
        Label binarizer
    test_data : pandas.DataFrame
        Test data with column names
    feature_name : str
        Name of the feature based on which slices are taken. Only categorical
        features allowed.

    Returns
    ------

    """

    # Process the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test_data, categorical_features=g_cat_features, label="salary",
        training=False, encoder=encoder, lb=label_binarizer
    )

    # Compute model predictions on the whole test data
    preds = inference_encoded(model, X_test)

    # Build groups for slicing
    feature_column = X_test[:, feature_no]
    groups = np.unique(feature_column)

    f = open('data/slice_output.txt', 'w')

    print(f"Model performance for feature no. {feature_no}:")

    for group in groups:

        group_idx = np.where(feature_column == group)[0]

        # y_test and preds should be binarized
        prec, rec, fbet = compute_model_metrics(
            y_test[group_idx], preds[group_idx])

        result_text = f"Slice {group} has precision: {prec}, recall: {rec}, "\
            f"fbeta: {fbet}"

        # Print result to stdout and file
        print(result_text)
        print(result_text, file=f)


def model_prediction(input_data):
    """
    Run an inference based on the trained model file and the given input data.

    Input
    -----
    X : pandas.DataFrame

    Returns
    -------
    predictions : np.array

    """

    # Load model and encoders
    model = load(g_fn_model)
    enc = load(g_fn_encoder)
    lb = load(g_fn_label_binarizer)

    # Make sure data has the same no of features as training data w/o the
    # label column
    assert input_data.shape[1] == 14

    X_enc, _, _, _ = process_data(
        input_data, categorical_features=g_cat_features, label=None,
        training=False, encoder=enc, lb=lb
    )

    predictions_bin = inference_encoded(model, X_enc)

    predictions = lb.inverse_transform(predictions_bin)

    return predictions


def run_pipeline():
    """
    Run the ml pipeline: data loading, preparation, training, testing, storing

    Inputs
    ------

    Returns
    -------

    """

    # Load data
    census_data = pd.read_csv('./data/census_clean.csv')

    # Prepare data
    train, test = train_test_split(census_data, test_size=0.20)
    # Optional enhancement, use K-fold cross validation instead of a
    # train-test split.

    # Train model
    model, enc, lb = model_training(train)
    
    # Evaluate overall model performance
    model_testing_overall(model, enc, lb, test)

    # Evaluate model performance on selected feature
    # Build processed feature list
    features = np.concatenate(
        [census_data.columns[~census_data.columns.isin(
            g_cat_features)].values, g_cat_features])
    model_testing_slices(
        model, enc, lb, test, np.where(features == 'sex')[0][0])

    # Save to file
    dump(model, g_fn_model)
    dump(enc, g_fn_encoder)
    dump(lb, g_fn_label_binarizer)


if __name__ == "__main__":

    run_pipeline()

#     # Load data
#     census_data = pd.read_csv('./data/census_clean.csv')
#
#     # Run pipeline
#     model = run_ml_pipeline(census_data)
#
#     # Save model
#     dump(model, './model/rf_model.joblib')
