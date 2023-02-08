import cv2 as cv
import mediapipe as mp
import time

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

# turn on camera
cap = cv.VideoCapture(1)
while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # hands only uses RGB images
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:  #detects hands
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):  #landmark (lm) gives ratio of x and y pos // id is point on hand (i.e. 0 for wrist)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)  #x and y position in pixels
                print(id, cx, cy)

                cv.circle(img, (cx, cy), 20, (255,0,255), cv.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  #plots points on hands

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img,str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)  #fps tracker

    cv.imshow("Image", img)
    cv.waitKey(1)
