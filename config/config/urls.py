from django.contrib import admin
from django.urls import path, include

app_name = 'chatbot'

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('chatbot.urls')), # アクセス先のアプリを設定
]
