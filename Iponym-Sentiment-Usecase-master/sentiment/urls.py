"""Iponym360 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
     path('model_evaluation/model_metrics', views.model_metrics),
     path('model_evaluation/sentiment_analysis', views.sentiment),
     path('model_evaluation/reviews',views.reviews),
     path('model_evaluation/model_comparison', views.models_comparison),
     path('model_evaluation/search_keyword', views.get_keyword),
     path('data_statistics/about_data',views.data_stats),
     path('data_statistics/wordcloud', views.word_cloud),
     path('data_statistics/age', views.display_age),
     path('data_statistics/ratings', views.display_rating),
     path('data_statistics/recommended', views.get_recommended),
     path('data_statistics/class', views.display_class_name),
 ]
