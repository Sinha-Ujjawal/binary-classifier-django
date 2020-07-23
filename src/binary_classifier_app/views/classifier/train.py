from rest_framework.views import APIView, Response
from rest_framework import serializers
from ...binary_classifier import BinaryClassifier
from ...services import create_binary_classifier, save_dataset
from ...base_logger import BASE_LOGGER


LOGGER = BASE_LOGGER.getChild("views.classifier.train")


class Train(APIView):
    class TrainInputSerializer(serializers.Serializer):
        plotPoints = serializers.ListField(
            help_text="""provide 2d data in an array in the form:
                [
                    [[x1, y1], [x2, y2], ...], # type 0
                    [[x3, y3], [x4, y4], [x5, y5], ...] # type 1
                ]
            """
        )
        modelType = serializers.CharField(default="dummy")
        modelArgs = serializers.DictField(default={})

    def post(self, request, format=None):
        train_input_serializer = self.TrainInputSerializer(data=request.data)
        train_input_serializer.is_valid(raise_exception=True)

        type_0, type_1 = train_input_serializer.validated_data["plotPoints"]
        model_type = train_input_serializer.validated_data["modelType"]
        model_args = train_input_serializer.validated_data["modelArgs"]

        LOGGER.info(f"Training model on datapoints: type_0: {type_0}, type_1: {type_1}")
        LOGGER.info(f"Model type: {model_type}")
        LOGGER.info(f"Model Args: {model_args}")
        model = BinaryClassifier(
            model_type=model_type, model_args=model_args, type_0=type_0, type_1=type_1,
        )
        binary_classifier = create_binary_classifier(model=model, is_training=False)
        LOGGER.info(
            f"Model training and saved to db, with model_id: {binary_classifier.model_id}!"
        )

        LOGGER.info("Saving training data for future reference ...")
        save_dataset(binary_classifier=binary_classifier, type_0=type_0, type_1=type_1)
        LOGGER.info("Training data saved!")

        return Response(
            {
                "modelId": binary_classifier.model_id,
                "modelType": model_type,
                "modelArgs": model_args,
            }
        )
