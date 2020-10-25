from django.urls import path
from .views import *

urlpatterns = [
    path('ensemble-classification', EnsembleClassification.as_view()),
]