from django.core.exceptions import ObjectDoesNotExist
from rest_framework.views import APIView, Response
from ...base_logger import BASE_LOGGER
from ...models import BinaryClassifierModel

LOGGER = BASE_LOGGER.getChild("views.classifier.delete_model")


class DeleteModel(APIView):
    def delete(self, request, model_id: int, format=None):
        status = None
        try:
            LOGGER.info(f"Deleting model with model_id: {model_id} ...")
            binary_classifier = BinaryClassifierModel.objects.get(model_id=model_id)
            binary_classifier.delete()
            LOGGER.info("Model deleted!")
            status = "DELETED"
        except ObjectDoesNotExist:
            status = "NOT-FOUND"
        except:
            status = "ERROR-OCCURED"

        return Response({"modelId": model_id, "status": status})
