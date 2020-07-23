from django.test import TestCase
from ..binary_classifier import BinaryClassifier
from ..services import create_binary_classifier, save_dataset
from ..selectors import get_all_model_ids, get_training_data
from ..base_logger import BASE_LOGGER

LOGGER = BASE_LOGGER.getChild("tests.test_models")


def save_dummy_classifier():
    dummy_model = BinaryClassifier(
        model_type="logistic", model_args={}, type_0=[[1, 2]], type_1=[[4, 5]]
    )
    return create_binary_classifier(model=dummy_model)


class BinaryClassifierModelTests(TestCase):
    def setUp(self):
        self.model_id = save_dummy_classifier().model_id

    def test_can_save_sklearn_model(self):
        self.assertEquals(get_all_model_ids(), [self.model_id])


class TrainingDataModelTests(TestCase):
    type_0 = type_1 = sorted([[0.45, 0.78], [4.56, 1.345], [4.5, -6.7]])

    def setUp(self):
        self.model = save_dummy_classifier()
        save_dataset(
            binary_classifier=self.model, type_0=self.type_0, type_1=self.type_1
        )

    def test_can_store_data(self):
        training_data = get_training_data()

        type_0 = []
        type_1 = []
        for data in training_data:
            X = [float(data.x), float(data.y)]
            if data.label == 0:
                type_0.append(X)
            else:
                type_1.append(X)

        type_0 = sorted(type_0)
        type_1 = sorted(type_1)

        self.assertEqual(type_0, self.type_0)
        self.assertEqual(type_1, self.type_1)
