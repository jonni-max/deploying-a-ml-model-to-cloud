"""
test_model.py

Test functions related to the model training and testing pipeline.
"""

import pytest
import numpy as np
import sklearn.ensemble

from model.ml.model import train_model, compute_model_metrics
from model.ml.model import inference_encoded


pytest.x_data = np.array([[0, 1, 0], [0, 0, 0], [1, 1, 1]])
pytest.y_data = np.array([1, 0, 1])


def pytest_configure():
    # TODO seems to be not called by pytest
    #     pytest.x_data = np.random.rand(3,2)
    #     pytest.y_data = np.random.rand(3,1)
    pass


def test_train_model():

    assert pytest.y_data is not None

    model = train_model(pytest.x_data, pytest.y_data)

    assert isinstance(model, sklearn.ensemble.RandomForestClassifier)


def test_compute_model_metrics():

    #     prediction = np.random.rand(3,1)
    prediction = np.array([1, 1, 1])

    precision, recall, fbeta = compute_model_metrics(pytest.y_data, prediction)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_inference_encoded():

    model = train_model(pytest.x_data, pytest.y_data)

    X = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])

    prediction = inference_encoded(model, X)

    assert prediction.shape == pytest.y_data.shape


if __name__ == "__main__":

    #     pytest_configure()

    test_train_model()
    test_compute_model_metrics()
    test_inference_encoded()
