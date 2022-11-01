import cv2
import numpy as np
import cvzone

nose_cascade = cv2.CascadeClassifier('Haar_Classifier/haarcascade_mcs_nose.xml')

if nose_cascade.empty():
  raise IOError('Unable to load the nose cascade classifier xml file')

cap = cv2.VideoCapture(0)
ds_factor = 1.0

overlay = cv2.imread('images/rednose.png', cv2.IMREAD_UNCHANGED)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in nose_rects:
        #cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        overlay_resize = cv2.resize(overlay,(w,h))
        frame = cvzone.overlayPNG(frame, overlay_resize,[int(x),int(y)])
        break

    cv2.imshow('Nose Detector', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()