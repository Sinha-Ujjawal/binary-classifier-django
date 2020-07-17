from rest_framework.views import APIView, Response
from ...models import BinaryClassifierModel


class TestBinaryClassifierView(APIView):
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

        binary_classifier = BinaryClassifierModel.objects.get(model_id=model_id)
        label = binary_classifier.model.predict([[x, y]])[0]

        return Response({**json_data, **{"label": label}})
