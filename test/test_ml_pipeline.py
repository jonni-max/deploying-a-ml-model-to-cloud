"""
Test for the ml pipeline

"""


import pandas as pd
# import sklearn.ensemble

# from ..model.ml.model import train_model, compute_model_metrics, inference
from model.ml_pipeline import model_prediction


def test_model_prediction():

    # Load data
    census_data = pd.read_csv('data/census_test.csv')

    x_data = census_data.drop(['salary'], axis=1).iloc[1:3, :]

    y_out = model_prediction(x_data)

    print(y_out)

    assert len(y_out) == x_data.shape[0]
