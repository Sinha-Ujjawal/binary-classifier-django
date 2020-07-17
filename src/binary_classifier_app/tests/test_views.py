from django.test import TransactionTestCase
from ..base_logger import BASE_LOGGER
from rest_framework.test import APIRequestFactory
from ..models import TrainingDataModel
from ..views import TrainBinaryClassifierView
from ..views import TestBinaryClassifierView

LOGGER = BASE_LOGGER.getChild("tests.test_views")


class TrainTestBinaryClassifierViewTest(TransactionTestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        request = self.factory.post(
            "/train/",
            {"plotPoints": [[[0, 1], [1, 1]], [[0, 3], [4, 5], [1, 2]]]},
            format="json",
        )
        response = TrainBinaryClassifierView.as_view()(request)
        print(response.data)
        self.model_id = response.data["modelId"]

    def test_can_store_data(self):
        print(TrainingDataModel.objects.values())

    def test_can_test_after_train(self):
        request = self.factory.post(
            "/train/", {"modelId": self.model_id, "x": 0.5, "y": 0.5,}, format="json",
        )
        response = TestBinaryClassifierView.as_view()(request)
        print(response.data)
