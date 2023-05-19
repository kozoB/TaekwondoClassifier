import cv2
import mediapipe as mp
import time
import math

class poseDetector():
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=True, smooth_segmentation=True,
                  min_detection_confidence=0.5, min_tracking_confidence = 0.5):
        # self.mode = mode
        # self.upBody = upBody
        # self.smooth = smooth
        # self.detectionCon = detectionCon
        # self.trackCon = trackCon

        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.static_image_mode, model_complexity=self.model_complexity, smooth_landmarks=self.smooth_landmarks,
                                     enable_segmentation = self.enable_segmentation, smooth_segmentation = self.smooth_segmentation,
                                     min_detection_confidence = self.min_detection_confidence, min_tracking_confidence = self.min_tracking_confidence)

    def findPose(self, img, draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=False):
        self.lmList = []
    
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        return self.lmList
    
    def findAngle(self, img, point1, point2, point3, draw=True):
        # Get the landmarks
        x1, y1 = self.lmList[point1][1:]
        x2, y2 = self.lmList[point2][1:]
        x3, y3 = self.lmList[point3][1:]

        # Calculate angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        if angle <= 0:
            angle += 360

        if angle > 180:
            angle = 380 - angle
        #print(angle)

        # Draw
        if draw:
            cv2.circle(img, (x1, y1), 3, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 3, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 3, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 7, (0,0,255), 2)
            cv2.circle(img, (x2, y2), 7, (0,0,255), 2)
            cv2.circle(img, (x3, y3), 7, (0,0,255), 2)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        return angle
    
    def getDetections(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        return self.results
    

def main():
    cap = cv2.VideoCapture('vidoe.mp4')
    prevTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.getPosition(img)
        print(lmList)

        currTime = time.time()
        fps = 1/(currTime-prevTime)
        prevTime = currTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Video", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
