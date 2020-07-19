from rest_framework.views import APIView, Response
import numpy as np
from ...base_logger import BASE_LOGGER
from ...models import BinaryClassifierModel

LOGGER = BASE_LOGGER.getChild("views.classifier.decission_boundary")


def init_X():
    H = 100
    points = np.arange(0, 1000, H)
    xx, yy = np.meshgrid(points, points)
    X = np.c_[xx.ravel(), yy.ravel()]
    return X


X = init_X()


class DecisionBoundary(APIView):
    def post(self, request, format=None):
        # {
        #     "modelId":,
        # }
        json_data = request.data

        model_id = json_data["modelId"]

        LOGGER.info(f"Retrieving model for model_id: {model_id} ...")
        binary_classifier = BinaryClassifierModel.objects.get(model_id=model_id)
        LOGGER.info("Model retrieved!")

        LOGGER.info("Predicting the labels for the mesh")
        X_transformed = binary_classifier.standard_scaler.transform(X)
        labels = binary_classifier.model.predict(X_transformed)
        LOGGER.info("Labels predicted")

        type_0 = []
        type_1 = []
        for point, label in zip(X, labels):
            if label == 0:
                type_0.append(point)
            else:
                type_1.append(point)

        return Response({**json_data, **{"plotPoints": [type_0, type_1]}})
