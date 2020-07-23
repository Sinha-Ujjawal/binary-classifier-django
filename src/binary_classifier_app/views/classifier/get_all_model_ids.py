from rest_framework.views import APIView, Response
from ...selectors import get_all_model_ids
from ...base_logger import BASE_LOGGER

LOGGER = BASE_LOGGER.getChild("views.classifier.get_all_model_ids")


class ModelIds(APIView):
    def get(self, request, format=None):
        LOGGER.info("Retrieving all model_ids from db")
        model_ids = get_all_model_ids()
        LOGGER.info("Models ids retrieved!")

        return Response({"modelIds": model_ids})
