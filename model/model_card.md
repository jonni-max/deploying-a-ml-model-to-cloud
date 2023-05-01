# Model Card

For additional information see the Model Card paper: <https://arxiv.org/pdf/1810.03993.pdf>

## Model Details
The model is a random forest classifier with 100 trees and gini impurity as criterion function.

## Intended Use
We want to predict peoples salary based on information about their lifestyle.

## Training Data
We trained the model with the census dataset of the UCI Machine Learning Repository.

<https://archive.ics.uci.edu/ml/datasets/census+income>

## Evaluation Data
We used 20 % of the census dataset for testing, ie. evaluation of the model.

## Metrics
We extracted model metrics, precision, recall and fbeta on sclices of male/female:

Male: precision: 0.7423510466988728, recall: 0.6150767178118746, fbeta: 0.6727471725647574
Female: precision: 0.8113207547169812, recall: 0.6142857142857143, fbeta: 0.6991869918699187

## Ethical Considerations
We observed a bias in the training data, since precision of the trained model varies significantly between male and female groups. 

## Caveats and Recommendations
We do not guarantee for correctness of the model predictions.