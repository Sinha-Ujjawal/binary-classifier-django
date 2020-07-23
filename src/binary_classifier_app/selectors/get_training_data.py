from ..models import TrainingDataModel


def get_training_data(*, model_id: int = None):
    if model_id:
        return TrainingDataModel.objects.filter(binary_classifier_model=model_id).all()
    else:
        return TrainingDataModel.objects.all()
