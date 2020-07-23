from rest_framework.views import APIView, Response
from rest_framework import serializers
from ...selectors import get_binary_classifier_model
from ...base_logger import BASE_LOGGER


LOGGER = BASE_LOGGER.getChild("views.classifier.test")


class Test(APIView):
    class TestInputSerializer(serializers.Serializer):
        modelId = serializers.IntegerField()
        x = serializers.DecimalField(decimal_places=4, max_digits=10)
        y = serializers.DecimalField(decimal_places=4, max_digits=10)

    def post(self, request, format=None):
        test_input_serializer = self.TestInputSerializer(data=request.data)
        test_input_serializer.is_valid(raise_exception=True)

        model_id = test_input_serializer.validated_data["modelId"]
        x = test_input_serializer.validated_data["x"]
        y = test_input_serializer.validated_data["y"]

        LOGGER.info(f"Retrieving model for model_id: {model_id} ...")
        model = get_binary_classifier_model(model_id=model_id).model
        LOGGER.info("Model retrieved!")

        LOGGER.info(f"Predicting the label for data point: x: {x}, y: {y}")
        X = [[x, y]]
        X_transfomed = model.scaler.transform(X)
        label = model.clf.predict(X_transfomed)[0]
        LOGGER.info(f"Label predicted: {label}")

        return Response({"modelId": model_id, "x": x, "y": y, "label": label,})
