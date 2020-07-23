from django.core.exceptions import ObjectDoesNotExist
from rest_framework.views import APIView, Response
from ...services import delete_binary_classifier
from ...base_logger import BASE_LOGGER

LOGGER = BASE_LOGGER.getChild("views.classifier.delete_model")


class DeleteModel(APIView):
    def delete(self, request, model_id: int, format=None):
        status = None
        try:
            LOGGER.info(f"Deleting model with model_id: {model_id} ...")
            delete_binary_classifier(model_id=model_id)
            LOGGER.info("Model deleted!")
            status = "DELETED"
        except ObjectDoesNotExist:
            status = "NOT-FOUND"
        except:
            status = "ERROR-OCCURED"

        return Response({"modelId": model_id, "status": status})
