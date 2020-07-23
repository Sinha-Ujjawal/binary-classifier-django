from ..models import BinaryClassifierModel


def get_binary_classifier_model(*, model_id: int) -> BinaryClassifierModel:
    binary_classifier = BinaryClassifierModel.objects.get(model_id=model_id)
    return binary_classifier
