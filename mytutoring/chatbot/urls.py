# chatbot/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat, name='chat'),  # Endpoint for chatbot interactions
]
