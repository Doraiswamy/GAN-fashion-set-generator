from django.urls import path
from .views import *

urlpatterns = [
    path('gan-generator', GANGenerator.as_view()),
]