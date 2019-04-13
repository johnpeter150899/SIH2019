from django.shortcuts import render
from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from .pro import getPPTDetails1,ppttolist,clust
import os


class design(TemplateView):
    template_name = "index.html"


def index(request):
        return render(request, 'index.html')


def clustering(request):
    search_item={}
    if request.method == 'POST':
        search = request.POST['search_clust']
        search_item['data']=clust(search)
    return render(request, 'clustering.html',search_item)

def about(request):
    return render(request, 'about.html')    

def download(request):
    path = 'media/'
    url = [[x,'/'+path+x.replace(' ','%20')] for x in os.listdir(path) if x.endswith(".pptx")]
    return render(request, 'download.html',{'urls':url})  


def index2(request):
    search_item={}
    if request.method == 'POST':
        search = request.POST['search']
        #det='sample'
        directory = [x for x in os.listdir('media/') if x.endswith(".pptx")]
        if len(directory)>0:
            det = getPPTDetails1(search)
            search_item = {'search':det}
        else :
            search_item['message']='NO FILES FOUND'
    return render(request, 'index2.html',search_item)


def upload(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)
        ppttolist(uploaded_file.name)
    return render(request, 'upload11.html', context)


def upload2(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        if uploaded_file.name.endswith(".pptx"):
            fs = FileSystemStorage()
            path = 'media/'
            directory = [x for x in os.listdir(path) if x.endswith(".pptx")]
            if uploaded_file.name in directory:
                context['message']='File Already Exists'
            else:
                name = fs.save(uploaded_file.name, uploaded_file)
                context['url'] = fs.url(name)
                ppttolist(uploaded_file.name)
                context['message']='File Uploaded Successfully'
        else:
            context['message']='The File is not in the PPT format'
                    
    return render(request, 'upload2.html', context)



def uploadpdf(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)
        ppttolist(uploaded_file.name)
    return render(request, 'uploadpdf.html', context)


def uploadpdf2(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        path = 'media/'
        directory = [x for x in os.listdir(path) if x.endswith(".pptx")]
        if uploaded_file.name in directory:
            context['message']='File Already Exists'
        else:
            name = fs.save(uploaded_file.name, uploaded_file)
            context['url'] = fs.url(name)
            ppttolist(uploaded_file.name)
            context['message']='File Uploaded Successfully'
    return render(request, 'upload.html', context)
