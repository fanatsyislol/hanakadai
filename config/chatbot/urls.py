from django.urls import path
from . import views

app_name = 'chatbot'

urlpatterns = [
        path('', views.home, name='home'),
        path('bot_response/', views.bot_response, name='bot_response'), # 応答
]