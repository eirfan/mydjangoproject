from django.urls import path
from .views import StudentApiView,DetailStudentApiView


urlpatterns = [
    path('',StudentApiView.as_view()),
    path('<int:pk>/',DetailStudentApiView.as_view())
]