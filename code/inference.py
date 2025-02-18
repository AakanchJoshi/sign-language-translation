import numpy as np
import pandas as pd
import os
import cv2
import time
import pickle
import mediapipe as mp
from tensorflow import keras
from csv import writer
import sqlite3

# Database connection
conn = sqlite3.connect('SignLanguage.db')
c = conn.cursor()

# Load actions
with open('actions.pkl', 'rb') as file:
    actions = pickle.load(file)

# Load sign language model
model = keras.models.load_model('sign_detection.h5')

# Media pipe Initialization
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    # Draw the pose landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def prob_viz(res, actions, input_frame):
    output_frame = input_frame.copy()
    top_5_idx = np.argsort(res)[-3:]
    top_5_prob = [res[i] for i in top_5_idx]
    act = [actions[id] for id in top_5_idx]

    for num, prob in enumerate(top_5_prob):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), (255,0,0), -1)
        cv2.putText(output_frame, act[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)

    return output_frame

def getUniqueWords(sentence):
    #list which contains unique words
    unique_words = []
    for s in sentence[-15:]:
        if s not in unique_words:
            unique_words.append(s)

    return unique_words

def prepareSentence(unique_words):
    sent_prediction = []
    if len(unique_words) > 6:
        last_few_unique_words = unique_words[-7:]
        sent_for_pred = ' '.join(last_few_unique_words)
        sent_prediction.append(sent_for_pred)

    return sent_prediction

def insertToDB(sent_prediction):
    if not sent_prediction:
        pass
    else:
        c.executemany("INSERT INTO sentences VALUES (?)", [sent_prediction])
        conn.commit()

# Variables
sequence = []
sentence = []
threshold = 0.5
sent_prediction = []
sent = ""
static_label = "Predicted Sentence:"
sent_bool = False

# Opencv video capture
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read frames
        ret, frame = cap.read()
        # Make keypoints detection
        image, results = mediapipe_detection(frame, holistic)
        # Draw landmarks
        # draw_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]

            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            # image = prob_viz(res, actions, image)

        unique_words = getUniqueWords(sentence)
        sent_prediction = prepareSentence(unique_words)
        insertToDB(sent_prediction)

        if not sent_prediction:
            pass
        else:
            try:
                c.execute("SELECT rowid, sent FROM predicted_sentence ORDER BY rowid DESC LIMIT 1")
                sent = c.fetchone()[1]
                conn.commit()
                sent_bool = True
            except:
                pass

        if sent_bool:
            cv2.putText(image, static_label, (3,30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(image, sent, (3,50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 0), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('Frames', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()
