from django.urls import path
from . import views

urlpatterns = [
    path("train", views.Train.as_view()),
    path("test", views.Test.as_view()),
    path("decisionBoundary/<int:model_id>", views.DecisionBoundary.as_view()),
    path("models", views.ModelIds.as_view()),
    path("deleteModel/<int:model_id>", views.DeleteModel.as_view()),
]
