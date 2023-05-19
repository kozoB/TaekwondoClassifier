"""
VIDEO INPUT: Video input, change to whatever is used - input_data = 'mixed.mp4'
RECORDED LABEL: class_name = "Kick" - Change according to the recorded label in the video
RECORD_NEW_DATA: Record new data into the csv file for the model to train on - RECORD_NEW_DATA (True/False)
CREATE AND EVALUATE MODEL: create_model() - CREATE_MODDEL (True/False)
PREDICT NEW DATA USING SAVED MODEL: make_prediction() - MAKE_PREDICTION (True/False)
"""

import cv2
import mediapipe as mp
import time
import PoseModule as pm
import math

import csv
import numpy as np
import os.path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle

# PRE RUNNING PARAMETERS
RECORD_NEW_DATA = False # Flag if going to record new data (using the video in 'input_data' variable)
CREATE_MODEL = False # Flag if going to create model on recorded data in csv file
MAKE_PREDICTION = True # Flag if going to make prediction on input video
input_data = 'mixed.mp4' # Input video which will be recorded/used for prediction
class_name = "Idle" # Recorded class (label for the recorded data which will be present at the first column of the recorded rows in the csv file)

# Create a csv file in which data will be stored (only if it doesn't already exist), add column properties
def data_proccessing():
    if os.path.exists("coords.csv"):
        return

    # TODO new parts of data preproccessing
    results = detector.getDetections(img) #   TODO
    num_coords = len(results.pose_landmarks.landmark) # number of coordinates of each landmark
    landmarks = ['class']
    # Iterate through each landmark coordinate and add it to landmarks list
    for val in range(1, num_coords+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
    # Add the angles of the limbs too
    landmarks += ['ang_ra{}'.format(num_coords+1), 'ang_la{}'.format(num_coords+1), 'ang_rl{}'.format(num_coords+1), 'ang_ll{}'.format(num_coords+1)]
    # Write the list of coordinates and angles to a csv file
    with open ('coords.csv', mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)

# Append data to the csv file, define the current label being recorded as 'class_name'
def collect_data_flow():
    # TODO MAY BE THE SAME AS img RETURNED FROM img = detector.findPose(img)
    results = detector.getDetections(img) #   TODO
    pose = results.pose_landmarks.landmark

    # Store all pose landmark coordinates in a list variable 
    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
    # Store all limbs angles in a list variable 
    angle_row = list(np.array([rightArmAngle, leftArmAngle, rightLegAngle, leftLegAngle]).flatten())
    # Create a new list combining both lists
    row = pose_row + angle_row

    # Append class name to the first column
    row.insert(0, class_name)

    # Export data to coords.csv file
    with open ('coords.csv', mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(row)

# Create models, evaluate them and save the best model
def create_model():
    df = pd.read_csv('coords.csv')
    X = df.drop('class', axis=1) # features
    y = df['class'] # lables

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

    pipelines = {
        'lr':make_pipeline(StandardScaler(), LogisticRegression()),
        'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
        'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
        'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier())
    }

    fit_models = {}
    for algo, pipeline in pipelines.items():
        print(X_train)
        print(y_train)
        model = pipeline.fit(X_train, y_train)
        fit_models[algo] = model

    # Evaluate model
    for algo, model in fit_models.items():
        y_hat = model.predict(X_test)
        print(algo, accuracy_score(y_test, y_hat))

    # Save best model
    with open('attack_action.pk1', 'wb') as f:
        pickle.dump(fit_models['lr'], f)

# Make prediction with saved model on new data
def make_prediction():
    # Load saved model
    with open('attack_action.pk1', 'rb') as f:
        model = pickle.load(f)

    # Preproccess data for new prediction
    results = detector.getDetections(img) #   TODO
    """ RECORDED CLASS """
    pose = results.pose_landmarks.landmark

    # Store all pose landmark coordinates in a list variable 
    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
    # Store all limbs angles in a list variable 
    angle_row = list(np.array([rightArmAngle, leftArmAngle, rightLegAngle, leftLegAngle]).flatten())
    # Create a new list combining both lists
    row = pose_row + angle_row

    # Make predictions
    X = pd.DataFrame([row])
    attack_action_class = model.predict(X)[0]
    attack_action_prob = model.predict_proba(X)[0]
    print(attack_action_class, attack_action_prob)
    return attack_action_class, attack_action_prob

""" MODEL CREATION """
if CREATE_MODEL:
    create_model() # TODO MODEL CREATION
print("debug - create model")

""" VIDEO INPUT """
cap = cv2.VideoCapture("Videos\\" + input_data) # TODO CHANGE ACCORDING TO VIDEO INPUT

prevTime = 0
detector = pm.poseDetector()

angles = []
# Store "mp pose PoseLandmark" into a variable for repeated uses
landmark = mp.solutions.pose.PoseLandmark

while cap.isOpened:
    success, img = cap.read()
    try: 
        img = detector.findPose(img)
        lmList = detector.getPosition(img)
    except:
        break
    
    try:
        data_proccessing()
    except:
        pass

    if len(lmList) > 0:
        # Get angles of limbs
        rightArmAngle = detector.findAngle(img, landmark.RIGHT_SHOULDER.value, landmark.RIGHT_ELBOW, landmark.RIGHT_WRIST)
        leftArmAngle = detector.findAngle(img, landmark.LEFT_SHOULDER, landmark.LEFT_ELBOW, landmark.LEFT_WRIST)
        rightLegAngle = detector.findAngle(img, landmark.RIGHT_HIP, landmark.RIGHT_KNEE, landmark.RIGHT_ANKLE)
        leftLegAngle = detector.findAngle(img, landmark.LEFT_HIP, landmark.LEFT_KNEE, landmark.LEFT_ANKLE)

        # Left and right index fingers
        #left_hand_index_x1, left_hand_index_y1 = lmList[19][1], lmList[19][2]
        #right_hand_index_x2, right_hand_index_y2 = lmList[20][1], lmList[20][2]

        #cv2.circle(img, (left_index_x1, left_index_y1), 7, (255, 0, 255), cv2.FILLED)
        #cv2.circle(img, (right_index_x2, right_index_y2), 7, (255, 0, 255), cv2.FILLED)

        # Left and right index fingers
        #left_foot_index_x1, left_foot_index_y1 = lmList[31][1], lmList[31][2]
        #right_foot_index_x2, right_foot_index_y2 = lmList[32][1], lmList[32][2]

        #cv2.circle(img, (left_foot_index_x1, left_index_y1), 7, (255, 0, 255), cv2.FILLED)
        #cv2.circle(img, (right_foot_index_x2, right_index_y2), 7, (255, 0, 255), cv2.FILLED)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # Predict new data using saved model
    if MAKE_PREDICTION:
        try:
            prediction, probability = make_prediction()
            cv2.putText(img, f'Pred: {str(prediction)}', (30, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            if prediction == "Idle":
                probability = probability[0]
            elif prediction == "Kick":
                probability = probability[1]
            else:
                probability = probability[2]
            cv2.putText(img, f'Prob: {probability:.2f}', (30, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        except:
            print('No data to predict')

    cv2.imshow("Video", img)
    cv2.waitKey(1)

    # Collect data of current video stream into csv file
    if RECORD_NEW_DATA:
        try:
            collect_data_flow()
        except:
            print("Frame with no data or csv file opened")

cap.release()
cv2.destroyAllWindows()



