from django.db import models
from .binary_classifier_model import BinaryClassifierModel


class TrainingDataModel(models.Model):
    def __str__(self):
        return str(
            {
                "binary_classifier_model": self.binary_classifier_model,
                "x": self.x,
                "y": self.y,
                "label": int(self.label),
                "crte_ts": self.crte_ts,
            }
        )

    binary_classifier_model = models.ForeignKey(
        BinaryClassifierModel,
        on_delete=models.CASCADE,
        parent_link=True,
        related_name="training_data",
    )
    x = models.DecimalField(decimal_places=4, max_digits=5)
    y = models.DecimalField(decimal_places=4, max_digits=5)
    label = models.BooleanField()
    crte_ts = models.DateTimeField(auto_now=True)
