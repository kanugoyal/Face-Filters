import cv2
import cvzone

cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier('Haar_Classifier/haarcascade_frontalface_default.xml')
overlay = cv2.imread('images/nose.png', cv2.IMREAD_UNCHANGED)
#print(overlay)
while True:
    _, frame = cap.read()
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray_scale)
    for (x, y, w , h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        
        overlay_resize = cv2.resize(overlay,(int(w/1.5),int(h/2)))
        frame = cvzone.overlayPNG(frame, overlay_resize,[x+30,y+50])

    cv2.imshow('Snap filter', frame)
    if cv2.waitKey(10) == ord('q'):
        break