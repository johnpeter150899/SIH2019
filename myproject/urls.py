"""myproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
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
from django.urls import path
from myapp import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('admin/', admin.site.urls),
    path('webpage/',views.index,name='index'),
    path('webpage2/',views.index2,name='index2'),
    path('upload/', views.upload,name='upload'),
    path('upload2/', views.upload2,name='upload2'),
    path('uploadpdf/', views.uploadpdf,name='uploadpdf'),
    path('uploadpdf2/', views.uploadpdf2,name='uploadpdf2'),
    path('about/', views.about ,name='about'),
    path('', views.index,),
    path('download/', views.download,name='download'),
    path('clustering/', views.clustering,name='clustering'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL,document_root=settings.STATIC_ROOT)
