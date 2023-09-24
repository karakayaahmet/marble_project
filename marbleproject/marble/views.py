
from django.shortcuts import render, redirect
from .models import Resimler
from .forms import ImageForm
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import statistics
import numpy
import cv2
    
def anasayfa(request):
    son_nesne = Resimler.objects.order_by("created_at").last()
    son_resim = son_nesne.image.url
    # ad = son_nesne.title
    # renk = son_nesne.average_color
    # crack = son_nesne.crack
    # dot = son_nesne.dot
    # good = son_nesne.good
    # joint = son_nesne.joint
    # kalite_turu = son_nesne.kalite_turu
    az_hasar = son_nesne.az_hasarli
    orta_hasar = son_nesne.orta_hasarli
    agir_hasar = son_nesne.agir_hasarli
    
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        
        if form.is_valid():

            form.save()
            
            return redirect("anasayfa")
         

    else:
        form = ImageForm()


    return render(request, "marble/anasayfa.html",{"images":son_resim, "form":form, "az_hasar":az_hasar, "orta_hasar":orta_hasar, "agir_hasar":agir_hasar})

