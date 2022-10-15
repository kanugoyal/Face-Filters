import cv2
import dlib
from imutils import face_utils

cap = cv2.VideoCapture(0)

while True:

        ret, frame = cap.read()
        #frame = cv2.resize(frame,(256,256))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detect = dlib.get_frontal_face_detector()

        rects = face_detect(gray, 1)
        
        for (i, rect) in enumerate(rects):

            (x, y, w, h) = face_utils.rect_to_bb(rect)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        #out.write(frame)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()