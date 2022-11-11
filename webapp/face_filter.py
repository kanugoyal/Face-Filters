import cv2
import numpy as np
from .filter_choice import filter_prep


def face_filter(img, val):
    
    cascPath = "Haar_Classifier/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    fil = filter_prep(val)

    while True:
        
        #read each frame of the video and convert it to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find faces in image using classifier
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40,40)
        )
        
        # For every face found 
        for (x,y,w,h) in faces:
            
            
            #dog filter
            if val == 1:

                face_width = w
                face_height = h
                
                fil = cv2.resize(fil, (int(face_width * 1), int(face_height * 0.4)))
                for i in range(int(face_height * 0.4)):
                    for j in range(int(face_width * 1)):
                        for k in range(3):
                            if fil[i][j][k] < 235:
                                img[y + i + int(0.25* h) - 1][x + j + int(0.005 * w)][k] = fil[i][j][k]
                 
            
            #glasses filter
            if val >= 2 and val <= 4:

                face_width = w
                face_height = h

                fil = cv2.resize(fil, (int(face_width * 1), int(face_height * 0.4)))
                for i in range(int(face_height * 0.4)):
                    for j in range(int(face_width * 1)):
                        for k in range(3):
                            if fil[i][j][k] < 235:
                                img[y + i + int(0.25* h) - 1][x + j + int(0.005 * w)][k] = fil[i][j][k]
                
            
            #spiderman & ironman
            if val == 5 and val == 6:

                face_width = w
                face_height = h

                fil = cv2.resize(fil, (int(face_width * 0.9), int(face_height * 1.4)))
                for i in range(int(face_height * 1.4)):
                    for j in range(int(face_width * 0.9)):
                        for k in range(3):
                            if fil[i][j][k] < 235:
                                img[y + i - int(0.3 * h) - 1][x + j + int(0.05 * w)][k] = fil[i][j][k]
                


            
        #display image
        return(img)
        