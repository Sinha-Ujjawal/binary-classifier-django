from rest_framework.views import APIView, Response
from ...base_logger import BASE_LOGGER
from ...models import BinaryClassifierModel

LOGGER = BASE_LOGGER.getChild("views.classifier.test")


class Test(APIView):
    def post(self, request, format=None):
        # {
        #     "modelId":,
        #     "x":
        #     "y":
        # }
        json_data = request.data

        model_id = json_data["modelId"]
        x = json_data["x"]
        y = json_data["y"]

        LOGGER.info(f"Retrieving model for model_id: {model_id} ...")
        binary_classifier = BinaryClassifierModel.objects.get(model_id=model_id)
        LOGGER.info("Model retrieved!")

        LOGGER.info(f"Predicting the label for data point: x: {x}, y: {y}")
        X = [[x, y]]
        X_transfomed = binary_classifier.standard_scaler.transform(X)
        label = binary_classifier.model.predict(X_transfomed)[0]
        LOGGER.info(f"Label predicted: {label}")

        return Response({**json_data, **{"label": label}})
