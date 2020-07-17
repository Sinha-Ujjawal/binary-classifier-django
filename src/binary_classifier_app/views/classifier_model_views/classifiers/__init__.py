from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

CLASSIFIER_STORE = {
    "dummy": DummyClassifier,
    "logistic": LogisticRegression,
    "naive bayes": GaussianNB,
}
