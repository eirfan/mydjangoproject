from django.urls import path
from .models import decisiontreeclassier
from .views import machinelearningview
from .views import GoogleApiView
urlpatterns = [
    path('',machinelearningview),
    path('googledriveapi/',GoogleApiView)

]