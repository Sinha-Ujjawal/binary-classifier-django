from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from ..utils import make_Xy
from .classifier_store import classifier_store


class BinaryClassifier:
    def __init__(
        self,
        *,
        model_type: str,
        model_args: Dict[str, Any],
        type_0: List[List[int]],
        type_1: List[List[int]]
    ):
        # instantiating ml model
        model = classifier_store(model_type, model_args)
        ##

        # std scaler instantiation
        std_scaler = StandardScaler()
        ##

        # create X, y pairs from input data
        X, y = make_Xy(type_0=type_0, type_1=type_1)
        ##

        X_transformed = std_scaler.fit_transform(X)

        # fitting the model to X, y pairs (supervised learning)
        self._clf = model.fit(X_transformed, y)
        self._std_scaler = std_scaler
        ##

    @property
    def clf(self) -> BaseEstimator:
        return self._clf

    @property
    def scaler(self) -> TransformerMixin:
        return self._std_scaler
