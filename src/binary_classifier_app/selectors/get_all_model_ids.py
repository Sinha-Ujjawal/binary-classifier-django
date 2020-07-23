from typing import List
from ..models import BinaryClassifierModel


def get_all_model_ids() -> List[int]:
    return list(BinaryClassifierModel.objects.values_list("model_id", flat=True))
