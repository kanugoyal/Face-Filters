import cv2
import cvzone

cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
noseCascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
overlay = cv2.imread('pig_nose.png', cv2.IMREAD_UNCHANGED)
print(overlay)
while True:
    _, frame = cap.read()
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray_scale)
    for (x, y, w , h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray_scale[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        nose = noseCascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in nose:
            #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

          overlay_resize = cv2.resize(overlay,(ew,eh))
          frame = cvzone.overlayPNG(frame, overlay_resize,[int(x+ex),int(y+ey)])

    cv2.imshow('Snap filter', frame)
    if cv2.waitKey(10) == ord('q'):
        break