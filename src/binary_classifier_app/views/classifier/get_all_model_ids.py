from rest_framework.views import APIView, Response
from ...base_logger import BASE_LOGGER
from ...models import BinaryClassifierModel

LOGGER = BASE_LOGGER.getChild("views.classifier.get_all_model_ids")


class ModelIds(APIView):
    def get(self, request, format=None):
        LOGGER.info("Retrieving all model_ids from db")
        model_ids = BinaryClassifierModel.objects.values_list("model_id", flat=True)
        LOGGER.info("Models ids retrieved!")

        return Response({"modelIds": model_ids})
