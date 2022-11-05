import cv2
import numpy as np
import math
import sys
import datetime as dt
from time import sleep
import os
import subprocess
import cvzone

#cascPath = "Haar_Classifier/haarcascade_frontalface_default.xml"

#faceCascade = cv2.CascadeClassifier(cascPath)

class FaceFilters:

    def __init__(self, filters):
        self.cascpath = "Haar_Classifier/haarcascade_frontalface_default.xml"
        self.face_Cascade = cv2.CascadeClassifier(self.cascpath)
        self.filters = filters 

    def applyHaarCascade(self, frame , faceCascade, scaleFact = 1.1,   minNeigh = 5, minSizeW = 40):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features = faceCascade.detectMultiScale(
            gray,
            scaleFactor=scaleFact,
            minNeighbors=minNeigh,
            minSize=(minSizeW, minSizeW)
        )
        return features

    def put_glasses_filter(self, glasses, fil, x, y, w, h):
        face_width = w
        face_height = h

        glasses = cv2.resize(glasses, (int(face_width * 1), int(face_height * 0.4)))
        for i in range(int(face_height * 0.4)):
            for j in range(int(face_width * 1)):
                for k in range(3):
                    if glasses[i][j][k] < 235:
                        fil[y + i + int(0.25* h) - 1][x + j + int(0.005 * w)][k] = glasses[i][j][k]
        return fil   

    def put_eye_filter(self, eyes, fil, x, y, w, h):
        face_width = w
        face_height = h

        eyes = cv2.resize(eyes, (int(face_width * 1), int(face_height * 0.4)))
        for i in range(int(face_height * 0.4)):
            for j in range(int(face_width * 1)):
                for k in range(3):
                    if eyes[i][j][k] < 235:
                        fil[y + i + int(0.25* h) - 1][x + j + int(0.005 * w)][k] = eyes[i][j][k]
        return fil

    def put_eyelasses_filter(eyelasses, fil, x, y, w, h):
        face_width = w
        face_height = h

        eyelasses = cv2.resize(eyelasses, (int(face_width * 1), int(face_height * 0.4)))
        for i in range(int(face_height * 0.4)):
            for j in range(int(face_width * 1)):
                for k in range(3):
                    if eyelasses[i][j][k] < 235:
                        fil[y + i + int(0.25* h) - 1][x + j + int(0.005 * w)][k] = eyelasses[i][j][k]
        return fil
    
    def put_3dglasses_filter(tdglasses, fil, x, y, w, h):
        face_width = w
        face_height = h

        tdglasses = cv2.resize(tdglasses, (int(face_width * 1), int(face_height * 0.4)))
        for i in range(int(face_height * 0.4)):
            for j in range(int(face_width * 1)):
                for k in range(3):
                    if tdglasses[i][j][k] < 235:
                        fil[y + i + int(0.25* h) - 1][x + j + int(0.005 * w)][k] = tdglasses[i][j][k]
        return fil

    def put_swag_filter(swag, fil, x, y, w, h):
        face_width = w
        face_height = h

        swag = cv2.resize(swag, (int(face_width * 1), int(face_height * 0.4)))
        for i in range(int(face_height * 0.4)):
            for j in range(int(face_width * 1)):
                for k in range(3):
                    if swag[i][j][k] < 235:
                        fil[y + i + int(0.25* h) - 1][x + j + int(0.005 * w)][k] = swag[i][j][k]
        return fil

    def put_cat_filter(cat,fil,x,y,w,h):
        face_width = w
        face_height = h
        
        cat = cv2.resize(cat,(int(face_width*1.5),int(face_height*1.75)))
        for i in range(int(face_height*1.75)):
            for j in range(int(face_width*1.5)):
                for k in range(3):
                    if cat[i][j][k]<235:
                        fil[y+i-int(0.375*h)-1][x+j-int(0.25*w)][k] = cat[i][j][k]
        return fil

    def put_monkey_filter(monkey,fil,x,y,w,h):
        face_width = w
        face_height = h
        
        monkey = cv2.resize(monkey,(int(face_width*1.5),int(face_height*1.75)))
        for i in range(int(face_height*1.75)):
            for j in range(int(face_width*1.5)):
                for k in range(3):
                    if monkey[i][j][k]<235:
                        fil[y+i-int(0.375*h)-1][x+j-int(0.25*w)][k] = monkey[i][j][k]
        return fil

    def put_rabbit_filter(rab,fil,x,y,w,h):
        face_width = w
        face_height = h
        
        rab = cv2.resize(rab,(int(face_width*1.5),int(face_height*1.75)))
        for i in range(int(face_height*1.75)):
            for j in range(int(face_width*1.5)):
                for k in range(3):
                    if rab[i][j][k]<235:
                        fil[y+i-int(0.375*h)-1][x+j-int(0.25*w)][k] = rab[i][j][k]
        return fil

    def put_moustache1(mst1,fil,x,y,w,h):
    
        face_width = w
        face_height = h

        mst_width = int(face_width*0.4166666)+1
        mst_height = int(face_height*0.142857)+1



        mst1 = cv2.resize(mst1,(mst_width,mst_height))

        for i in range(int(0.62857142857*face_height),int(0.62857142857*face_height)+mst_height):
            for j in range(int(0.29166666666*face_width),int(0.29166666666*face_width)+mst_width):
                for k in range(3):
                    if mst1[i-int(0.62857142857*face_height)][j-int(0.29166666666*face_width)][k] <235:
                        fil[y+i][x+j][k] = mst1[i-int(0.62857142857*face_height)][j-int(0.29166666666*face_width)][k]
        return fil

    def put_moustache2(mst2,fil,x,y,w,h):
    
        face_width = w
        face_height = h

        mst_width = int(face_width*0.4166666)+1
        mst_height = int(face_height*0.142857)+1



        mst2 = cv2.resize(mst2,(mst_width,mst_height))

        for i in range(int(0.62857142857*face_height),int(0.62857142857*face_height)+mst_height):
            for j in range(int(0.29166666666*face_width),int(0.29166666666*face_width)+mst_width):
                for k in range(3):
                    if mst2[i-int(0.62857142857*face_height)][j-int(0.29166666666*face_width)][k] <235:
                        fil[y+i][x+j][k] = mst2[i-int(0.62857142857*face_height)][j-int(0.29166666666*face_width)][k]
        return fil

    def put_ironman_filter(ironman, fil, x, y, w, h):
        face_width = w
        face_height = h

        ironman = cv2.resize(ironman, (int(face_width * 0.9), int(face_height * 1.4)))
        for i in range(int(face_height * 1.4)):
            for j in range(int(face_width * 0.9)):
                for k in range(3):
                    if ironman[i][j][k] < 235:
                        fil[y + i - int(0.3 * h) - 1][x + j + int(0.05 * w)][k] = ironman[i][j][k]
        return fil

    def put_spiderman_filter(spiderman, fil, x, y, w, h):
        face_width = w
        face_height = h

        spiderman = cv2.resize(spiderman, (int(face_width * 0.9), int(face_height * 1.4)))
        for i in range(int(face_height * 1.4)):
            for j in range(int(face_width * 0.9)):
                for k in range(3):
                    if spiderman[i][j][k] < 235:
                        fil[y + i - int(0.3 * h) - 1][x + j + int(0.05 * w)][k] = spiderman[i][j][k]
        return fil

    def put_batman(bat, fil, x, y, w, h):
        face_width = w
        face_height = h

        bat_width = face_width + 1
        bat_height = int(0.35 * face_height) + 1

        bat = cv2.resize(bat, (bat_width, bat_height))

        for i in range(bat_height):
            for j in range(bat_width):
                for k in range(3):
                    if bat[i][j][k] < 235:
                        fil[y + i - int(0.25 * face_height)][x + j][k] = bat[i][j][k]
        return fil

    def put_capAmerica(cap, fil, x, y, w, h):
        face_width = w
        face_height = h

        cap_width = face_width + 1
        cap_height = int(0.35 * face_height) + 1

        cap = cv2.resize(cap, (cap_width, cap_height))

        for i in range(cap_height):
            for j in range(cap_width):
                for k in range(3):
                    if cap[i][j][k] < 235:
                        fil[y + i - int(0.25 * face_height)][x + j][k] = cap[i][j][k]
        return fil
    
