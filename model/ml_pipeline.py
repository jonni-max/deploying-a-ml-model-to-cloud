"""
Script to train and test the machine learning model on the census dataset.

"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from joblib import dump, load

import numpy as np
import pandas as pd

# Add the necessary imports for the starter code.
from .ml.data import process_data
from .ml.model import train_model, inference
from .ml.model import compute_model_metrics


def run_ml_pipeline(data):
    """
    Runs the components of the ml pipeline: train-test splitting, data processing,
    and training.
    
    Inputs
    ------
    data : pandas.DataFrame
        Input data for training and testing   
    
    Returns
    -------
    model : np.array
        Model trained with the input data
    
    """
    
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)
    
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, enc, lbin = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    
    # Process the test data with the process_data function.    
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder=enc, lb=lbin
    )
    
    # Train and save a model.
    model = train_model(X_train, y_train)
    
    # Build processed feature list
    features = np.concatenate([data.columns[~data.columns.isin(cat_features)].values, cat_features])
    
    # Evaluate model performance
    model_performance_slices(model, X_test, y_test, np.where(features=='sex')[0][0])
    
    return model


def model_performance_slices(model, X, y_label, feature_no):
    """
    Evaluate model performance on slices of one feature of the data.
    
    Input
    -----
    model : sklearn.ensemble.RandomForestClassifier
        Model to be evaluated    
    feature_no : int
        The index of the column of the feature based on which slices are taken. Only 
        categorical features allowed.
    y_label : np.array, 1d
        Known labels, binarized
    X : np.array, 1d
        Model input data
    
    Returns
    ------
    
    """

    # Compute model predictions on the whole dataset
    preds = inference(model, X)
    
    # Build groups for slicing
    feature_column = X[:,feature_no]
    groups = np.unique(feature_column)
    
    print(f"Model performance for feature no. {feature_no}:")
    
    for group in groups:
        
        group_idx = np.where(feature_column == group)[0]
        
        # y_label and preds should be binarized
        prec, rec, fbet = compute_model_metrics(y_label[group_idx], preds[group_idx])
                
        print(f"Slice {group} has precision: {prec}, recall: {rec}, fbeta: {fbet}")
    
    
#     # Take slices. Use pandas functionality
#     X_df = pd.DataFrame(X)
#     
#     x_slices = X_df.groupby[feature].groups
#     
#     print(f"Model performance for feature {feature}:")
#     
#     for slice_name in x_slices.keys():
#         
#         slice_idx = X_df.iloc[x_sclices[slice_name]].values
#         
#         # y_label and preds should be binarized
#         prec, rec, fbet = compute_model_metrics(y_label[slice_idx], preds[slice_idx])
#                 
#         print(f"Slice {slice_name} has precision: {prec}, recall: {rec}, fbeta: {fbet}")
        


if __name__ == "__main__":
    
    # Load data
    census_data = pd.read_csv('./data/census_clean.csv')
    
    # Run pipeline
    model = run_ml_pipeline(census_data)
    
    # Save model
    dump(model, './model/rf_model.joblib')
    
    
    