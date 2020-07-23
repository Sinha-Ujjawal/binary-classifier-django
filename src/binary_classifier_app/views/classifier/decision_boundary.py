from django.core.exceptions import ObjectDoesNotExist
from rest_framework.views import APIView, Response
import numpy as np
from ...selectors import get_binary_classifier_model
from ...base_logger import BASE_LOGGER

LOGGER = BASE_LOGGER.getChild("views.classifier.decission_boundary")


def init_X():
    H = 100
    points = np.arange(0, 1000, H)
    xx, yy = np.meshgrid(points, points)
    X = np.c_[xx.ravel(), yy.ravel()]
    return X


X = init_X()


class DecisionBoundary(APIView):
    def get(self, request, model_id: int, format=None):
        LOGGER.info(f"Retrieving model for model_id: {model_id} ...")
        model = get_binary_classifier_model(model_id=model_id).model
        LOGGER.info("Model retrieved!")

        LOGGER.info("Predicting the labels for the mesh")
        X_transformed = model.scaler.transform(X)
        labels = model.clf.predict(X_transformed)
        LOGGER.info("Labels predicted")

        type_0 = []
        type_1 = []
        for point, label in zip(X, labels):
            if label == 0:
                type_0.append(point)
            else:
                type_1.append(point)

        return Response({"modelId": model_id, "plotPoints": [type_0, type_1]})
