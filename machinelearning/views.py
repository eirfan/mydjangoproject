
from django.shortcuts import render
from .models import decisiontreeclassier
from .models import GoogleDriveApi
from django.http import HttpResponse
# Create your views here.

def machinelearningview(request):
    response = decisiontreeclassier()
    return HttpResponse(response)

def GoogleApiView(request):
    response = GoogleDriveApi()
    return HttpResponse(response) 