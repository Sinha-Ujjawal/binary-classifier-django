from django.test import TestCase
from rest_framework.test import APIRequestFactory
from ..views import Train, Test, ModelIds, DeleteModel
from ..selectors import get_training_data
from ..base_logger import BASE_LOGGER

LOGGER = BASE_LOGGER.getChild("tests.test_views")

api_request_factory = APIRequestFactory()


class TrainBinaryClassifierViewTests(TestCase):
    def setUp(self):
        request = api_request_factory.post(
            "classifier/train",
            {
                "modelType": "logistic",
                "modelArgs": {},
                "plotPoints": [[[0, 1], [1, 1]], [[0, 3], [4, 5], [1, 2]]],
            },
            format="json",
        )
        self.response = response = Train.as_view()(request)

    def test_can_train(self):
        assert "modelId" in self.response.data
        self.assertNotEqual(self.response.data["modelId"], None)


class TestBinaryClassifierViewTests(TrainBinaryClassifierViewTests):
    def test_can_test(self):
        request = api_request_factory.post(
            "classifier/test",
            {"modelId": self.response.data["modelId"], "x": 1.5, "y": 5.6,},
            format="json",
        )
        response = Test.as_view()(request)
        assert "label" in response.data
        assert response.data["label"] in {0, 1}


class GetAllModelIdsViewTests(TestCase):
    N_ITER = 10

    def setUp(self):
        for _ in range(self.N_ITER):
            request = api_request_factory.post(
                "classifier/train",
                {
                    "modelType": "logistic",
                    "modelArgs": {},
                    "plotPoints": [[[0, 1], [1, 1]], [[0, 3], [4, 5], [1, 2]]],
                },
                format="json",
            )
            Train.as_view()(request)

    def test_can_get_all_model_ids(self):
        request = api_request_factory.get("classifier/models")
        response = ModelIds.as_view()(request)
        self.assertEquals(len(response.data["modelIds"]), self.N_ITER)


class DeleteModelViewTests(TrainBinaryClassifierViewTests):
    def test_can_delete_model(self):
        model_id = self.response.data["modelId"]
        request = api_request_factory.delete(f"classifier/delete/{model_id}")
        DeleteModel.as_view()(request)
        # training_data = get_training_data(model_id=model_id)
        # self.assertEqual(len(training_data), 0)
