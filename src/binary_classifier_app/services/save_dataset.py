import itertools as it
from typing import List
from ..models import BinaryClassifierModel, TrainingDataModel


def save_dataset(
    *,
    binary_classifier: BinaryClassifierModel,
    type_0: List[List[int]],
    type_1: List[List[int]],
) -> bool:
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

    return True
