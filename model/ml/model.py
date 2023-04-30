from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

# Optional: implement hyperparameter tuning.


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier()  # default settings only

    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, encoder, lb, X):
    """
    Run model inferences and return the predictions using given model,
    encoder, label binarizer and data. Input and output are NOT encoded.

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        One hot encoder
    lb : sklearn.preprocessing._label.LabelBinarizer
        Label binarizer
    X : np.array
        Data (not encoded) used for prediction.

    Returns
    -------
    preds : np.array
        Predictions (not encoded) from the model.

    """

    X_enc = encoder.transform(X)  # encode input

    preds_enc = model.predict(X_enc)  # infer encoded output

    preds = lb.inverse_transform(preds_enc)  # decode output

    return preds


def inference_encoded(model, X_enc):
    """
    Run model inference on ENCODED input data.

    Input
    -----
    X_enc : np.array
        Encoded input data

    Returns
    -------
    pred_bin : np.array
        Predictions from the model, binarized

    """

    preds_bin = model.predict(X_enc)

    return preds_bin
