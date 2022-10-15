import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_y = 0

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):     #getting height and width of hand
                x = int(landmark.x*frame_width)
                y = int(landmark.y*frame_width)
                #print(x,y)
                if id == 8:    #becoz index finger tip is number 8
                    cv2.circle(img = frame, center=(x,y), radius = 20, color= (0,255,255) )
                    index_x = screen_width/frame_width*x
                    index_y = screen_height/frame_height*y
                    pyautogui.moveTo(index_x,index_y)
                if id == 4:   #for thumb is number 8
                    cv2.circle(img = frame, center=(x,y), radius = 20, color= (0,255,255) )
                    thumb_x = screen_width/frame_width*x
                    thumb_y = screen_height/frame_height*y
                    print('out', abs(index_y - thumb_y))
                    if abs(index_y - thumb_y) < 40:
                        pyautogui.click()
                        pyautogui.sleep(3)

    cv2.imshow('VIRTUAL MOUSE', frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break