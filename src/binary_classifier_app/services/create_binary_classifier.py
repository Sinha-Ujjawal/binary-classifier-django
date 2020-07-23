from ..binary_classifier import BinaryClassifier
from ..models import BinaryClassifierModel


def create_binary_classifier(
    *, model: BinaryClassifier, is_training: bool = True,
) -> BinaryClassifierModel:
    binary_clf = BinaryClassifierModel(model=model, is_training=is_training)
    binary_clf.save()
    return binary_clf
