from django.db import models
from picklefield.fields import PickledObjectField


class BinaryClassifierModel(models.Model):
    model_id = models.IntegerField(primary_key=True, db_index=True,)
    model = PickledObjectField(default=None)

    is_training = models.BooleanField(default=True)
    is_deleted = models.BooleanField(default=False)

    crte_ts = models.DateTimeField(auto_now=True)
