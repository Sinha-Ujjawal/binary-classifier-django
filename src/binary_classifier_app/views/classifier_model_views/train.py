from rest_framework.views import APIView, Response
import itertools as it
from typing import List, Tuple
from .classifiers import DummyClassifier
from ...models import BinaryClassifierModel, TrainingDataModel


class TrainBinaryClassifierView(APIView):
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
        type_0: List[List[int]],
        type_1: List[List[int]],
        *model_args,
        **model_kwargs
    ) -> BinaryClassifierModel:
        model = DummyClassifier(*model_args, **model_kwargs)
        X, y = self.make_Xy(type_0=type_0, type_1=type_1)
        model = model.fit(X, y)
        binary_classifier = BinaryClassifierModel(model=model, is_training=False)
        binary_classifier.save()
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
        # "plotPoints": [
        #     [[x, y]...], # Square
        #     [[x, y]...], # Circle
        # ]
        type_0, type_1 = request.data["plotPoints"]
        binary_classifier = self.create_binary_classifier(type_0=type_0, type_1=type_1)
        self.create_dataset(binary_classifier, type_0, type_1)
        return Response({"modelId": binary_classifier.model_id})
