from django.test import TransactionTestCase
from ..base_logger import BASE_LOGGER
from ..models import BinaryClassifierModel
from ..models import TrainingDataModel

LOGGER = BASE_LOGGER.getChild("tests.test_models")


def save_dummy_classifier() -> BinaryClassifierModel:
    from sklearn.dummy import DummyClassifier

    dummy_model = DummyClassifier()
    obj = BinaryClassifierModel(model=dummy_model, is_training=False, is_deleted=True,)
    obj.save()
    return obj


class BinaryClassifierModelTest(TransactionTestCase):
    def setUp(self):
        save_dummy_classifier()

    def test_can_save_sklearn_model(self):
        print(BinaryClassifierModel.objects.all())


class TrainingDataModelTest(TransactionTestCase):
    def setUp(self):
        self.model = model = save_dummy_classifier()
        points = [[0.45, 0.78], [4.56, 1.345], [4.5, -6.7]]
        TrainingDataModel.objects.bulk_create(
            map(
                lambda point: TrainingDataModel(
                    binary_classifier_model=model, x=point[0], y=point[1], label=True,
                ),
                points,
            )
        )

    def test_can_store_data(self):
        print(TrainingDataModel.objects.all())

    def test_can_model_see_data(self):
        print(self.model.training_data.all())
