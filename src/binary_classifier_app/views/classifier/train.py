from rest_framework.views import APIView, Response
import itertools as it
from typing import List, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from ...classifiers import classifier_store
from ...base_logger import BASE_LOGGER
from ...models import BinaryClassifierModel, TrainingDataModel


LOGGER = BASE_LOGGER.getChild("views.classifier.train")


class Train(APIView):
    def make_Xy(
        self, type_0: List[List[int]], type_1: List[List[int]]
    ) -> Tuple[List[List[int]], List[int]]:
        X = []
        y = []
        for label, coords in enumerate([type_0, type_1]):
            for coord in coords:
                X.append(coord)
                y.append(label)
        return X, y

    def create_binary_classifier(
        self,
        model_type: str,
        model_args: Dict[str, Any],
        type_0: List[List[int]],
        type_1: List[List[int]],
    ) -> BinaryClassifierModel:
        # instantiating ml model
        model = classifier_store(model_type, model_args)
        ##

        # std scaler instantiation
        std_scaler = StandardScaler()
        ##

        # create X, y pairs from input data
        X, y = self.make_Xy(type_0=type_0, type_1=type_1)
        ##

        X_transformed = std_scaler.fit_transform(X)

        # fitting the model to X, y pairs (supervised learning)
        model = model.fit(X_transformed, y)
        ##

        # instantiating and saving the classifier (ml model)
        # to the database
        binary_classifier = BinaryClassifierModel(
            model=model, standard_scaler=std_scaler, is_training=False
        )
        binary_classifier.save()
        ##

        return binary_classifier

    def create_dataset(
        self,
        binary_classifier: BinaryClassifierModel,
        type_0: List[List[int]],
        type_1: List[List[int]],
    ):
        create_map = lambda points, label: map(
            lambda point: TrainingDataModel(
                binary_classifier_model=binary_classifier,
                x=point[0],
                y=point[1],
                label=label,
            ),
            points,
        )

        TrainingDataModel.objects.bulk_create(
            it.chain(create_map(type_0, False), create_map(type_1, True))
        )

    def post(self, request, format=None):
        type_0, type_1 = request.data["plotPoints"]
        model_type = request.data.get("modelType", "dummy")
        model_args = request.data.get("modelArgs", {})

        LOGGER.info(f"Training model on datapoints: type_0: {type_0}, type_1: {type_1}")
        LOGGER.info(f"Model type: {model_type}")
        LOGGER.info(f"Model Args: {model_args}")
        binary_classifier = self.create_binary_classifier(
            model_type=model_type, model_args=model_args, type_0=type_0, type_1=type_1
        )
        LOGGER.info(
            f"Model training and saved to db, with model_id: {binary_classifier.model_id}!"
        )

        LOGGER.info("Saving training data for future reference ...")
        self.create_dataset(binary_classifier, type_0, type_1)
        LOGGER.info("Training data saved!")

        return Response(
            {
                "modelId": binary_classifier.model_id,
                "modelType": model_type,
                "modelArgs": model_args,
            }
        )
