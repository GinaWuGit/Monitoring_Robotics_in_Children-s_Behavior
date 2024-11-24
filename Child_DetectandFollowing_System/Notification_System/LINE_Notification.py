import cv2
import mediapipe as mp

import csv
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score  
import pickle
import requests
#------------------------------------------------------------------
mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic

df = pd.read_csv('coords_project.csv')

x = df.drop('class', axis=1) #features
y = df['class'] #target value

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)


pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier())
}

pipelines.keys()

list(pipelines.values())[1]

fit_models = {}
for algo, pipeline in pipelines.items():    
    model = pipeline.fit(x_train, y_train)  #train
    fit_models[algo] = model

fit_models['rc'].predict(x_test)



for algo, model in fit_models.items():
    yhat = model.predict(x_test)
    print(algo, accuracy_score(y_test, yhat))

fit_models['rf'].predict(x_test)  #以x_test數值去model預測動作

with open('pose_estimation_project.pkl', 'wb') as f: #train model
    pickle.dump(fit_models['rf'], f)

#-------------------------line notify設定-------------------------------------------------------
import requests

# 設置IFTTT Webhook URL和LINE Notify 權杖
ifttt_webhook_url = "https://maker.ifttt.com/trigger/【孩童不良行為發生!!】/with/key/cEHIyo8xraNi0CXMcoeocM"
line_notify_api_url = "https://notify-api.line.me/api/notify"
line_notify_access_token = "Q4srdncUDJdUGRcAljr2EyVSwp2iAygfWQwXbDSANVh"

# 要發送的訊息
message = "Gina"
message_thumb = '偵測到吸手指'
message_smile = '偵測到smile'
message_eyes = '偵測到揉眼睛'


# HTTP 標頭參數與資料
headers = { "Authorization": "Bearer " + token }
data_thumb = { 'message': message_thumb }
data_smile = { 'message': message_smile }
data_eyes = { 'message': message_eyes }


#--------------------------------------------------------------------------------

import time
start_time = time.time()

#--------------------------------------------------------------------------------

cap = cv2.VideoCapture(1)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False      
        
        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        
        #Export coordinates
        try:
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten()) #.flatten => 將多維變成一維陣列
            
            # Extract Face landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten()) #ex:[[1,2], [3,4]] => [1,2,3,4]
            
            # Concate row
            row = pose_row+face_row   
            
            #Make Detections
            X = pd.DataFrame([row])
            pose_estimation_class = model.predict(X)[0]
            pose_estimation_prob = model.predict_proba(X)[0]
            print(pose_estimation_class, pose_estimation_prob)

#-----------------------------偵測動作和計時發送訊息------------------------------------------            

            #輸出偵測到的動作
            print("pose_estimation_class的值:" + pose_estimation_class)
            if(pose_estimation_class == 'thumb'):
                print('這是thumb')
            if(pose_estimation_class == 'smile'):
                print('這是smile')
            if(pose_estimation_class == 'eyes'):
                print('這是eyes')

            #輸出偵測到的準確度
            print("pose_estimation_prob的值", round(pose_estimation_prob[np.argmax(pose_estimation_prob)],2))

            #如果準確度大於0.9且間隔超過3秒就發訊息
            elapsed_time = time.time() - start_time
            if((round(pose_estimation_prob[np.argmax(pose_estimation_prob)],2) > 0.9) and (elapsed_time > 3)):
                if(pose_estimation_class == 'thumb'):
                    print('這是thumb')
                    requests.post(ifttt_webhook_url.format(event='【孩童不良行為發生!!】', your_key='cEHIyo8xraNi0CXMcoeocM'), json={"value1": message, "value2": data_thumb})
                if(pose_estimation_class == 'smile'):
                    print('這是smile')
                    requests.post(ifttt_webhook_url.format(event='【孩童不良行為發生!!】', your_key='cEHIyo8xraNi0CXMcoeocM'), json={"value1": message, "value2": data_smile})
                if(pose_estimation_class == 'eyes'):
                    print('這是eyes')
                    requests.post(ifttt_webhook_url.format(event='【孩童不良行為發生!!】', your_key='cEHIyo8xraNi0CXMcoeocM'), json={"value1": message, "value2": data_eyes})

                #重製起算時間，繼續計時
                start_time = time.time()
#--------------------------------------------------------------------------------------------------            
            #Grab ear coords
            coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                          , [640,480]).astype(int))
            
            cv2.rectangle(image, 
                          (coords[0], coords[1]+5), 
                          (coords[0]+len(pose_estimation_class)*20, coords[1]-30), 
                          (245, 117, 16), -1)
            cv2.putText(image, pose_estimation_class, coords, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Get status box(左上)
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)  #(image, top-corner, bottom-corner, color, line-thinkness)
            
            # Display Class
            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, pose_estimation_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            
            
            # Display Probability
            cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(pose_estimation_prob[np.argmax(pose_estimation_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 
           
            print(str(round(pose_estimation_prob[np.argmax(pose_estimation_prob)])))
    
        except:
                pass    
 
        #cTime = time.time() #cTime = 現在的時間
        #fps = 1/(cTime-pTime)
        #pTime = cTime
        #cv2.putText(image, f"FPS : {int(fps)}", (480, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)    
    
        cv2.imshow('Holistic', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
            
cap.release()
cv2.destroyAllWindows()