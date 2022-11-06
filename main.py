import cv2
import numpy as np
import math
import sys
import logging as log
import datetime as dt
from time import sleep
import os
import subprocess

cascPath = "Haar_Classifier\haarcascade_frontalface_default.xml"  # for face detection

if not os.path.exists(cascPath):
    subprocess.call('./download_filters.sh', shell=True)
else:
    print('Filters already exist!')

faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log', level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0
mst = cv2.imread('images/moustache.png')
hat = cv2.imread('images/cowboy_hat.png')
dogFilter = cv2.imread('images/dog_filter.png')

cat_e = cv2.imread('images/cat_ear.jpg')
dog_e = cv2.imread('images/dog_e.jpg')
cat_n = cv2.imread('images/cat_n.jpg')
dog_n = cv2.imread('images/dog_n.jpg')
spiderman = cv2.imread('images/spiderman.png')
glasses = cv2.imread('images/glasses.jpg')
hearts = cv2.imread('images/hearts.jpg')


def put_moustache(mst, fc, x, y, w, h):
    face_width = w
    face_height = h

    mst_width = int(face_width * 0.4166666) + 1
    mst_height = int(face_height * 0.142857) + 1

    mst = cv2.resize(mst, (mst_width, mst_height))

    for i in range(int(0.62857142857 * face_height), int(0.62857142857 * face_height) + mst_height):
        for j in range(int(0.29166666666 * face_width), int(0.29166666666 * face_width) + mst_width):
            for k in range(3):
                if mst[i - int(0.62857142857 * face_height)][j - int(0.29166666666 * face_width)][k] < 235:
                    fc[y + i][x + j][k] = \
                        mst[i - int(0.62857142857 * face_height)][j - int(0.29166666666 * face_width)][k]
    return fc


def put_hat(hat, fc, x, y, w, h):
    face_width = w
    face_height = h

    hat_width = face_width + 1
    hat_height = int(0.35 * face_height) + 1

    hat = cv2.resize(hat, (hat_width, hat_height))

    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if hat[i][j][k] < 235:
                    fc[y + i - int(0.25 * face_height)][x + j][k] = hat[i][j][k]
    return fc


def put_dogFilter_filter(dogFilter, fc, x, y, w, h):
    face_width = w
    face_height = h

    dog = cv2.resize(dogFilter, (int(face_width * 1.5), int(face_height * 1.75)))
    for i in range(int(face_height * 1.75)):
        for j in range(int(face_width * 1.5)):
            for k in range(3):
                if dog[i][j][k] < 235:
                    fc[y + i - int(0.375 * h) - 1][x + j - int(0.25 * w)][k] = dog[i][j][k]
    return fc

def put_dog_filter(dog, fc, x, y, w, h):
    face_width = w
    face_height = h

    dog = cv2.resize(dog, (int(face_width * 1.5), int(face_height * 1.75)))
    for i in range(int(face_height * 1.75)):
        for j in range(int(face_width * 1.5)):
            for k in range(3):
                if dog[i][j][k] < 235:
                    fc[y + i - int(0.375 * h) - 1][x + j - int(0.25 * w)][k] = dog[i][j][k]
    return fc

def put_cat_e_filter(cat_e, fc, x, y, w, h):
    face_width = w
    face_height = h

    cat_e_width = face_width + 1
    cat_e_height = int(0.35 * face_height) + 1

    cat_e = cv2.resize(cat_e, (cat_e_width, cat_e_height))

    for i in range(cat_e_height):
        for j in range(cat_e_width):
            for k in range(3):
                if cat_e[i][j][k] < 235:
                    fc[y + i - int(0.25 * face_height)][x + j][k] = cat_e[i][j][k]
    return fc

def put_dog_e_filter(dog_e, fc, x, y, w, h):
    face_width = w
    face_height = h

    dog_e_width = face_width + 1
    dog_e_height = int(0.35 * face_height) + 1

    dog_e = cv2.resize(dog_e, (dog_e_width, dog_e_height))

    for i in range(dog_e_height):
        for j in range(dog_e_width):
            for k in range(3):
                if dog_e[i][j][k] < 235:
                    fc[y + i - int(0.25 * face_height)][x + j][k] = dog_e[i][j][k]
    return fc

def put_cat_n_filter(cat_n, fc, x, y, w, h):
    face_width = w
    face_height = h

    cat_n_width = int(face_width * 0.4166666) + 1
    cat_n_height = int(face_height * 0.142857) + 1

    cat_n = cv2.resize(cat_n, (cat_n_width, cat_n_height))

    for i in range(int(0.62857142857 * face_height), int(0.62857142857 * face_height) + cat_n_height):
        for j in range(int(0.29166666666 * face_width), int(0.29166666666 * face_width) + cat_n_width):
            for k in range(3):
                if cat_n[i - int(0.62857142857 * face_height)][j - int(0.29166666666 * face_width)][k] < 235:
                    fc[y + i][x + j][k] = \
                        cat_n[i - int(0.62857142857 * face_height)][j - int(0.29166666666 * face_width)][k]
    return fc

def put_dog_n_filter(dog_n, fc, x, y, w, h):
    face_width = w
    face_height = h

    dog_n_width = int(face_width * 0.4166666) + 1
    dog_n_height = int(face_height * 0.3) + 1

    dog_n = cv2.resize(dog_n, (dog_n_width, dog_n_height))

    for i in range(int(0.62857142857 * face_height), int(0.62857142857 * face_height) + dog_n_height):
        for j in range(int(0.29166666666 * face_width), int(0.29166666666 * face_width) + dog_n_width):
            for k in range(3):
                if dog_n[i - int(0.62857142857 * face_height)][j - int(0.29166666666 * face_width)][k] < 235:
                    fc[y + i][x + j][k] = \
                        dog_n[i - int(0.62857142857 * face_height)][j - int(0.29166666666 * face_width)][k]
    return fc

def put_spiderman_filter(spiderman, fc, x, y, w, h):
    face_width = w
    face_height = h

    spiderman = cv2.resize(spiderman, (int(face_width * 0.9), int(face_height * 1.4)))
    for i in range(int(face_height * 1.4)):
        for j in range(int(face_width * 0.9)):
            for k in range(3):
                if spiderman[i][j][k] < 235:
                    fc[y + i - int(0.3 * h) - 1][x + j + int(0.05 * w)][k] = spiderman[i][j][k]
    return fc

def put_glasses_filter(glasses, fc, x, y, w, h):
    face_width = w
    face_height = h

    glasses = cv2.resize(glasses, (int(face_width * 1), int(face_height * 0.4)))
    for i in range(int(face_height * 0.4)):
        for j in range(int(face_width * 1)):
            for k in range(3):
                if glasses[i][j][k] < 235:
                    fc[y + i + int(0.25* h) - 1][x + j + int(0.005 * w)][k] = glasses[i][j][k]
    return fc

def put_hearts_filter(hearts, fc, x, y, w, h):
    face_width = w
    face_height = h

    hearts = cv2.resize(hearts, (int(face_width * 1), int(face_height * 0.4)))
    for i in range(int(face_height * 0.4)):
        for j in range(int(face_width * 1)):
            for k in range(3):
                if hearts[i][j][k] < 235:
                    fc[y + i + int(0.25* h) - 1][x + j + int(0.005 * w)][k] = hearts[i][j][k]
    return fc


ch = 0

while (1):

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.putText(frame,"Person Detected",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        if ch == 2:
            frame = put_cat_e_filter(cat_e, frame, x, y, w, h)
            frame = put_cat_n_filter(cat_n, frame, x, y, w, h)
        elif ch == 1:
            frame = put_hearts_filter(hearts, frame, x, y, w, h)
        elif ch == 3:
            frame = put_spiderman_filter(spiderman, frame, x, y, w, h)
        elif ch == 4:
            frame = put_glasses_filter(glasses, frame, x, y, w, h)
        elif ch ==5:
            frame = put_dog_e_filter(dog_e, frame, x, y, w, h)
            frame = put_dog_n_filter(dog_n, frame, x, y, w, h)


    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: " + str(len(faces)) + " at " + str(dt.datetime.now()))

    # Display the resulting frame
    cv2.imshow('Video', frame)

    try:  # an error comes if it does not find anything in window as it cannot find contour of max area
        # therefore this try error statement

        frame = cv2.flip(frame, 1)
        kernel = np.ones((3, 3), np.uint8)

        # define region of interest
        roi = frame[100:300, 50:250]

        cv2.rectangle(frame, (50, 100), (250, 300), (255, 255, 255), 0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # define range of skin color in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # extract skin colur imagw
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask, kernel, iterations=4)

        # blur the image
        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        # find contours
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # find contour of max area(hand)
        cnt = max(contours, key=lambda x: cv2.contourArea(x))

        # approx the contour a little
        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # make convex hull around hand
        hull = cv2.convexHull(cnt)

        # define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)

        # find the percentage of area not covered by hand in convex hull
        arearatio = ((areahull - areacnt) / areacnt) * 100

        # find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)

        # l = no. of defects
        l = 0

        # code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt = (100, 180)

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            s = (a + b + c) / 2
            ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

            # distance between point and convex hull
            d = (2 * ar) / a

            # apply cosine rule here
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d > 30:
                l += 1
                cv2.circle(roi, far, 3, [255, 0, 0], -1)

            # draw lines around hand
            cv2.line(roi, start, end, [0, 255, 0], 2)

        l += 1

        # print corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame, '0.None 1.Hearts  2.Cat 3.Spider man 4.Glasses 5.Dog', (50, 450), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        if l == 1:
            if areacnt < 2000:
                cv2.putText(frame, 'Put hand in the box!', (0, 70), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
                ch = 0
            else:
                if arearatio < 12:
                    cv2.putText(frame, '0.None', (70, 70), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
                    ch = 0
                elif arearatio < 17.5:
                    # cv2.putText(frame, 'Best of luck', (0, 50), font, 1.5, (189, 186, 141), 3, cv2.LINE_AA)
                    cv2.putText(frame, '1.Hearts', (70, 70), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
                    ch = 1

                else:
                    cv2.putText(frame, '1.Hearts', (70, 70), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
                    ch = 1

        elif l == 2:
            cv2.putText(frame, '2.Cat', (70, 70), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
            ch = 2

        elif l == 3:

            if arearatio < 27:
                cv2.putText(frame, '3.Spider man', (70, 70), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
                ch = 3
            else:
                # cv2.putText(frame, 'ok', (140, 50), font, 2, (189, 186, 141), 3, cv2.LINE_AA)
                cv2.putText(frame, '3.Spider man', (70, 70), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
                ch = 3

        elif l == 4:
            cv2.putText(frame, '4.Glasses', (70, 70), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
            ch = 4

        elif l == 5:
            cv2.putText(frame, '5.Dog', (70, 70), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
            ch = 5

        elif l == 6:
            cv2.putText(frame, 'reposition!', (70, 70), font, 2, (255, 255, 255), 3, cv2.LINE_AA)

        else:
            cv2.putText(frame, 'reposition!', (700, 70), font, 2, (255, 255, 255), 3, cv2.LINE_AA)

        # show the windows
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)
    except:
        pass

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
video_capture.release()















