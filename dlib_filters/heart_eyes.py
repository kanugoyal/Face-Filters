import dlib
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
img = cv2.imread("images/heart.png")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert the video into gray for faster processing
    
    faces = detector(gray) #detect faces in the gray frame 
    for face in faces:
        
        a1 = face.left()
        b1 = face.top()
        a2 = face.right()
        b2 = face.bottom()
        faceHeight = b2 - b1
        faceWidth = a2- a1
        #print(faceHeight,faceWidth)
        #cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
        
        landmarks = predictor(gray, face)
        
        leftEyeLeft = (landmarks.part(36).x,landmarks.part(36).y)
        leftEyeRight = (landmarks.part(39).x,landmarks.part(39).y)
        rightEyeLeft = (landmarks.part(42).x,landmarks.part(42).y)
        rightEyeRight = (landmarks.part(45).x,landmarks.part(45).y)
        
        heartWidth = leftEyeRight[0] - leftEyeLeft[0]
        heartWidth = 3 * heartWidth
        heartHeight = heartWidth
        
        heartImage = cv2.resize(img, (heartWidth,heartHeight))
        heartImageGray = cv2.cvtColor(heartImage, cv2.COLOR_BGR2GRAY)
        
        leftEyeCenter = ((leftEyeLeft[0] + leftEyeRight[0])/2,(leftEyeLeft[1] + leftEyeRight[1])/2)
        leftTopLeft = (int(leftEyeCenter[0] - heartWidth / 2),int(leftEyeCenter[1] - heartHeight / 2))
        leftBottomRight = (int(leftEyeCenter[0] + heartWidth / 2),int(leftEyeCenter[1] + heartHeight / 2))
        
        rightEyeCenter = ((rightEyeLeft[0] + rightEyeRight[0])/2,(rightEyeLeft[1] + rightEyeRight[1])/2)
        rightTopLeft = (int(rightEyeCenter[0] - heartWidth / 2),int(rightEyeCenter[1] - heartHeight / 2))
        rightBottomRight = (int(rightEyeCenter[0] + heartWidth / 2),int(rightEyeCenter[1] + heartHeight / 2))        
        
        x1 = landmarks.part(62).x
        y1 = landmarks.part(62).y
        x2 = landmarks.part(66).x
        y2 = landmarks.part(66).y
        
        if (y2-y1)/faceHeight > 0.05:
            print("mouth open")
            heartAreaLeft = frame[leftTopLeft[1] : leftTopLeft[1] + heartHeight,
                              leftTopLeft[0] : leftTopLeft[0] + heartWidth]  
            _, heartMask = cv2.threshold(heartImageGray, 25, 255, cv2.THRESH_BINARY_INV)
            heartAreaNoHeartLeft = cv2.bitwise_and(heartAreaLeft,heartAreaLeft, mask = heartMask)
            finalHeartLeft = cv2.add(heartAreaNoHeartLeft,heartImage)
            frame[leftTopLeft[1] : leftTopLeft[1] + heartHeight,
                         leftTopLeft[0] : leftTopLeft[0] + heartWidth]  = finalHeartLeft
                  
            heartAreaRight = frame[rightTopLeft[1] : rightTopLeft[1] + heartHeight,
                              rightTopLeft[0] : rightTopLeft[0] + heartWidth]  
            heartAreaNoHeartRight = cv2.bitwise_and(heartAreaRight,heartAreaRight, mask = heartMask)
            finalHeartRight = cv2.add(heartAreaNoHeartRight,heartImage)
            frame[rightTopLeft[1] : rightTopLeft[1] + heartHeight,
                         rightTopLeft[0] : rightTopLeft[0] + heartWidth]  = finalHeartRight
                  
        
    cv2.imshow("frame", frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break