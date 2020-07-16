from django.test import TestCase
from ..models import BinaryClassifierModel

# Create your tests here.
class BinaryClassifierModelTest(TestCase):
    def test_can_store_sklearn_model(self):
        from sklearn.dummy import DummyClassifier

        dummy_model = DummyClassifier()
        obj = BinaryClassifierModel(
            model=dummy_model, is_training=False, is_deleted=True,
        )
        obj.save()
