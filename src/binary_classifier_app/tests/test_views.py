from django.test import TransactionTestCase
from ..base_logger import BASE_LOGGER
from rest_framework.test import APIRequestFactory
from ..models import TrainingDataModel
from ..views import Train
from ..views import Test

LOGGER = BASE_LOGGER.getChild("tests.test_views")

api_request_factory = APIRequestFactory()


class TrainBinaryClassifierView_Test(TransactionTestCase):
    def setUp(self):
        request = api_request_factory.post(
            "classifier/train/",
            {
                "modelType": "logistic",
                "modelArgs": {},
                "plotPoints": [[[0, 1], [1, 1]], [[0, 3], [4, 5], [1, 2]]],
            },
            format="json",
        )
        response = Train.as_view()(request)
        print(response.data)
        self.model_id = response.data["modelId"]

    def test_can_store_data(self):
        print(TrainingDataModel.objects.values())

    def test_can_test_after_train(self):
        request = api_request_factory.post(
            "classifier/test/",
            {"modelId": self.model_id, "x": 0.5, "y": 0.5,},
            format="json",
        )
        response = Test.as_view()(request)
        print(response.data)
