import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import numpy as np
import os
import subprocess
import cvzone

cascPath = "Haar_Classifier/haarcascade_frontalface_default.xml"  # for face detection

if not os.path.exists(cascPath):
    subprocess.call(['./download_filters.sh'])
else:
    print('Filters already exist!')

faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0
mst = cv2.imread('images/moustache.png')
hat = cv2.imread('images/cowboy_hat.png')
cat_e = cv2.imread('images/cat_ear.jpg')
cat_n = cv2.imread('images/cat_n.jpg')
dog_n = cv2.imread('images/dog_n.jpg')
dog = cv2.imread('images/dog_filter.png')
glasses = cv2.imread('images/glasses.jpg')
hearts = cv2.imread('images/hearts.jpg')
swag = cv2.imread('images/swag.png',  cv2.IMREAD_UNCHANGED)
spiderman = cv2.imread('images/spiderman.jpg')
ironman = cv2.imread('images/ironman.jpg')


def put_moustache(mst,fil,x,y,w,h):
    
    face_width = w
    face_height = h

    mst_width = int(face_width*0.4166666)+1
    mst_height = int(face_height*0.142857)+1



    mst = cv2.resize(mst,(mst_width,mst_height))

    for i in range(int(0.62857142857*face_height),int(0.62857142857*face_height)+mst_height):
        for j in range(int(0.29166666666*face_width),int(0.29166666666*face_width)+mst_width):
            for k in range(3):
                if mst[i-int(0.62857142857*face_height)][j-int(0.29166666666*face_width)][k] <235:
                    fil[y+i][x+j][k] = mst[i-int(0.62857142857*face_height)][j-int(0.29166666666*face_width)][k]
    return fil

def put_hat(hat,fil,x,y,w,h):
    
    face_width = w
    face_height = h
    
    hat_width = face_width+1
    hat_height = int(0.35*face_height)+1
    
    hat = cv2.resize(hat,(hat_width,hat_height))
    
    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if hat[i][j][k]<235:
                    fil[y+i-int(0.25*face_height)][x+j][k] = hat[i][j][k]
    return fil

def put_cat_ear(cat_e, fil, x, y, w, h):
    face_width = w
    face_height = h

    cat_e_width = face_width + 1
    cat_e_height = int(0.35 * face_height) + 1

    cat_e = cv2.resize(cat_e, (cat_e_width, cat_e_height))

    for i in range(cat_e_height):
        for j in range(cat_e_width):
            for k in range(3):
                if cat_e[i][j][k] < 235:
                    fil[y + i - int(0.25 * face_height)][x + j][k] = cat_e[i][j][k]
    return fil

def put_cat_nose(cat_n, fil, x, y, w, h):
    face_width = w
    face_height = h

    cat_n_width = int(face_width * 0.4166666) + 1
    cat_n_height = int(face_height * 0.142857) + 1

    cat_n = cv2.resize(cat_n, (cat_n_width, cat_n_height))

    for i in range(int(0.62857142857 * face_height), int(0.62857142857 * face_height) + cat_n_height):
        for j in range(int(0.29166666666 * face_width), int(0.29166666666 * face_width) + cat_n_width):
            for k in range(3):
                if cat_n[i - int(0.62857142857 * face_height)][j - int(0.29166666666 * face_width)][k] < 235:
                    fil[y + i][x + j][k] = \
                        cat_n[i - int(0.62857142857 * face_height)][j - int(0.29166666666 * face_width)][k]
    return fil

def put_dog_nose(dog_n, fil, x, y, w, h):
    face_width = w
    face_height = h

    dog_n_width = int(face_width * 0.4166666) + 1
    dog_n_height = int(face_height * 0.3) + 1

    dog_n = cv2.resize(dog_n, (dog_n_width, dog_n_height))

    for i in range(int(0.62857142857 * face_height), int(0.62857142857 * face_height) + dog_n_height):
        for j in range(int(0.29166666666 * face_width), int(0.29166666666 * face_width) + dog_n_width):
            for k in range(3):
                if dog_n[i - int(0.62857142857 * face_height)][j - int(0.29166666666 * face_width)][k] < 235:
                    fil[y + i][x + j][k] = \
                        dog_n[i - int(0.62857142857 * face_height)][j - int(0.29166666666 * face_width)][k]
    return fil

def put_dog_filter(dog,fil,x,y,w,h):
    face_width = w
    face_height = h
    
    dog = cv2.resize(dog,(int(face_width*1.5),int(face_height*1.75)))
    for i in range(int(face_height*1.75)):
        for j in range(int(face_width*1.5)):
            for k in range(3):
                if dog[i][j][k]<235:
                    fil[y+i-int(0.375*h)-1][x+j-int(0.25*w)][k] = dog[i][j][k]
    return fil
    
    
def put_swag_glasses(swag,fil,x,y,w,h):
    face_width = w
    face_height = h
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
    overlay_resize = cv2.resize(swag,(face_width,face_height))
    fil = cvzone.overlayPNG(frame, overlay_resize,[x,y-20])
    return fil

def put_glasses_filter(glasses, fil, x, y, w, h):
    face_width = w
    face_height = h

    glasses = cv2.resize(glasses, (int(face_width * 1), int(face_height * 0.4)))
    for i in range(int(face_height * 0.4)):
        for j in range(int(face_width * 1)):
            for k in range(3):
                if glasses[i][j][k] < 235:
                    fil[y + i + int(0.25* h) - 1][x + j + int(0.005 * w)][k] = glasses[i][j][k]
    return fil

def put_hearts_filter(hearts, fil, x, y, w, h):
    face_width = w
    face_height = h

    hearts = cv2.resize(hearts, (int(face_width * 1), int(face_height * 0.4)))
    for i in range(int(face_height * 0.4)):
        for j in range(int(face_width * 1)):
            for k in range(3):
                if hearts[i][j][k] < 235:
                    fil[y + i + int(0.25* h) - 1][x + j + int(0.005 * w)][k] = hearts[i][j][k]
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
  
ch = 0
print ("Select Filter: 1.) Hat 2.) Moustache 3.) Hat and Moustache 4.) Dog Filter 5.) Swag Filter 6.) Swag Filter and Hat 7.) Swag Filter and Moustache 8.) Cat Ear 9.) Cat Filter 10.) Dog Nose 12.) Spiderman 11.) Ironman")
ch = int(input())
    
    
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40,40)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.putText(frame,"Person Detected",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        
        if ch==1:
            frame = put_hat(hat,frame,x,y,w,h)
        elif ch==2:
            frame = put_moustache(mst,frame,x,y,w,h)
        elif ch==3:
            frame = put_moustache(mst,frame,x,y,w,h)
            frame = put_hat(hat,frame,x,y,w,h)
        elif ch==4:
            frame = put_dog_filter(dog,frame,x,y,w,h)
        elif ch==5:
            frame = put_swag_glasses(swag,x,y,w,h)
        elif ch==6:
            frame = put_swag_glasses(swag,x,y,w,h)
            frame = put_hat(hat,frame,x,y,w,h)
        elif ch==7 :
            frame = put_swag_glasses(swag,x,y,w,h)
            frame = put_moustache(mst,frame,x,y,w,h)
        elif ch==8 :
            frame = put_cat_ear(cat_e, frame, x, y, w, h)
        elif ch==9 :
            frame = put_cat_ear(cat_e, frame, x, y, w, h)
            frame = put_cat_nose(cat_n, frame, x, y, w, h)
        elif ch==10 :
            frame = put_dog_nose(dog_n, frame, x, y, w, h)
        elif ch==11 :
            frame = put_ironman_filter(ironman, frame, x, y, w, h)
        else :
            frame = put_spiderman_filter(spiderman, frame, x, y, w, h)
            
            
            
    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        break


    

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()