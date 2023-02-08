import cv2 as cv
import mediapipe as mp
import time


class handDetector:
    def __init__(self, mode=False, maxHands=2, modelCom=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelCom = modelCom
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelCom, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # hands only uses RGB images
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:  # detects hands
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)  # plots points on hands
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):  # landmark (lm) gives ratio of x and y pos / id is point on hand
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # x and y position in pixels
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 20, (255, 0, 255), cv.FILLED)

        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(1)  # turn on camera
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])  # prints 4th point coords

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  # fps tracker
        cv.imshow("Image", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
