import cv2
import dlib
from imutils import face_utils
import matplotlib.pyplot as plt
def detectDlib(imgpath):
    frame = cv2.imread(imgpath)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detect = dlib.get_frontal_face_detector()
    rects = face_detect(gray, 1)
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
    plt.imshow(frame)
    plt.show()


detectDlib('images\kids2.png')
detectDlib('images\people.jpg')
detectDlib('images\people1.jpg')