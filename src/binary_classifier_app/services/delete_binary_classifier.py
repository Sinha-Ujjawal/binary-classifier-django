from ..models import BinaryClassifierModel


def delete_binary_classifier(*, model_id: int) -> BinaryClassifierModel:
    binary_clf = BinaryClassifierModel.objects.get(model_id=model_id)
    binary_clf.delete()
    return binary_clf
