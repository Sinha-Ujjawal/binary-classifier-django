from django.db import models
from picklefield.fields import PickledObjectField


class BinaryClassifierModel(models.Model):
    def __str__(self):
        return str(
            {
                "model_id": self.model_id,
                "model": self.model,
                "is_training": self.is_training,
                "is_deleted": self.is_deleted,
                "crte_ts": self.crte_ts,
            }
        )

    model_id = models.AutoField(primary_key=True, db_index=True)
    model = PickledObjectField(default=None)

    is_training = models.BooleanField(default=True)
    is_deleted = models.BooleanField(default=False)

    crte_ts = models.DateTimeField(auto_now=True)
