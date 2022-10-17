import cv2
import dlib
import numpy as np

def mask(frame, landmarks):
    glasses = cv2.imread("images/swag.png", -1)
    orig_mask = glasses[:, :, 3]
    orig_mask_inv = cv2.bitwise_not(orig_mask)
    glasses = glasses[:, :, 0:3]
    origGlassesHeight, origGlassesWidth = glasses.shape[:2]

    glassesWidth = abs(3*(landmarks.part(37).x - landmarks.part(46).x))
    glassesHeight = int(glassesWidth*origGlassesHeight/origGlassesWidth)
    swag_glasses = cv2.resize(glasses, (glassesWidth, glassesHeight), interpolation= cv2.INTER_AREA)
    mask = cv2.resize(orig_mask, (glassesWidth, glassesHeight), interpolation= cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv, (glassesWidth, glassesHeight), interpolation= cv2.INTER_AREA)

    y1 = int(landmarks.part(27).y - (glassesHeight / 2))
    y2 = int(y1 + glassesHeight)
    x1 = int(landmarks.part(27).x - (glassesWidth / 2))
    x2 = int(x1 + glassesWidth)
    roi = frame[y1:y2, x1:x2]
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    roi_fg = cv2.bitwise_and(swag_glasses, swag_glasses, mask=mask)
    frame[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)

    return frame

def filter():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(frameGray)
            for face in faces:
                landmarks = predictor(frameGray, face)

                frame = mask(frame, landmarks)
            cv2.imshow('Detector', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    filter()