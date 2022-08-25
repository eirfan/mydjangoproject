from django.shortcuts import render

# Create your views here.
from student.models import Student
from machinelearning.models import decisiontreeclassier
from rest_framework import generics
from .serializers import StudentSerializer

class StudentApiView(generics.ListAPIView):
    queryset = Student.objects.all()
    serializer_class = StudentSerializer


class DetailStudentApiView(generics.RetrieveAPIView):
    # Student.objects.all will retrieve all the object @ data inside the database
    queryset = Student.objects.all()
    print(queryset)
    serializer_class = StudentSerializer
    print("read here")
