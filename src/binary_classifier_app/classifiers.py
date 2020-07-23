from typing import Dict, Any, Callable

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def init_classifier_store() -> Callable[[str, Dict[str, Any]], BaseEstimator]:
    classifier_factory = {
        "dummy": DummyClassifier,
        "logistic": LogisticRegression,
        "logisticCV": LogisticRegressionCV,
        "svm": SVC,
        "naive bayes": GaussianNB,
        "knn": KNeighborsClassifier,
        "rnn": RadiusNeighborsClassifier,
        "decission tree": DecisionTreeClassifier,
        "random forest": RandomForestClassifier,
        "neural network": MLPClassifier,
    }

    def apply(model_type: str, model_args: Dict[str, Any]) -> BaseEstimator:
        if model_type not in classifier_factory:
            raise Exception(
                f"Model type: {model_type} not found in classifier_factory, only {classifier_factory.keys()} are allowed!"
            )
        return classifier_factory[model_type](**model_args)

    return apply


classifier_store = init_classifier_store()
