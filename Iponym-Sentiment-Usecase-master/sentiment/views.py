from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from .Controller import entry_point
import json


# Create your views here.


def model_metrics(request):
    response = entry_point.get_model_metrics()
    return JsonResponse(response)


def sentiment(request):
    response = entry_point.getting_sentiment()
    return JsonResponse(response)


def reviews(request):
    response = entry_point.display_reviews()
    return HttpResponse(response)


def models_comparison(request):
    response = entry_point.get_model_comparison()
    return HttpResponse(json.dumps(response))


def get_keyword(request):
    var = request.GET.get('keyword')
    response = entry_point.search_by_keyword(var)
    return HttpResponse(response)


def data_stats(request):
    response = entry_point.get_data_stats()
    return JsonResponse(response)


def word_cloud(request):
    category = request.GET.get('product')
    response = entry_point.make_word_cloud(category)
    return HttpResponse(response)


def display_age(request):
    category = request.GET.get('product')
    response = entry_point.age(category)
    return HttpResponse(response)


def display_rating(request):
    category = request.GET.get('product')
    response = entry_point.ratings(category)
    return HttpResponse(response)


def get_recommended(request):
    category = request.GET.get('product')
    response = entry_point.recommended_items(category)
    return HttpResponse(response)


def display_recommended(request):
    category = request.GET.get('product')
    response = entry_point.recommended_items(category)
    return HttpResponse(response)


def display_class_name(request):
    response = entry_point.class_name()
    return HttpResponse(response)
