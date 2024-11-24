import cv2
import mediapipe as mp
import numpy as np
import math
import time
import serial


#建立連接阜
# ser = serial.Serial('COM10', 9600, timeout=1)
time.sleep(3)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 加載姿勢檢測模型
pose_detection = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 用於儲存頭身比的暫存區
buffer = []  

# 讀取照片
cap = cv2.VideoCapture(0)


while True:
    # fps detect
    start_time = time.time() # stasrt time of the loop
    
    # 讀取攝像頭中的影像
    ret, frame = cap.read()

    if not ret:
        print("無法讀取攝像頭影像")
        break

    # 進行姿勢檢測
    results = pose_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
        
        return int(angle)

    def calculate_distance(a, b):
        a = np.multiply(a, [frame.shape[1], frame.shape[0]]).astype(int)
        b = np.multiply(b, [frame.shape[1], frame.shape[0]]).astype(int)
    
        distance = math.hypot(b[0] - a[0], b[1] - a[1])
        return distance
        
    

    # 檢測到關鍵點才進行繪製和分析
    if results.pose_landmarks is not None:
        landmarks = results.pose_landmarks.landmark
        
        # 有檢測到關鍵點
        z = "有檢測到骨架"
        
        # 繪製關鍵點
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
        right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y]
        left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
        left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
        mid_eye = [(right_eye[0]+left_eye[0])/2, (right_eye[1]+left_eye[1])/2]
        mid_shoulder = [(right_shoulder[0]+left_shoulder[0])/2, (right_shoulder[1]+left_shoulder[1])/2]
        mid_heel = [(right_heel[0]+left_heel[0])/2, (right_heel[1]+left_heel[1])/2]
        mid_hip = [(right_hip[0]+left_hip[0])/2, (right_hip[1]+left_hip[1])/2]
        
        
        ratio = ((calculate_distance(mid_eye, nose)*3)/(calculate_distance(mid_shoulder, mid_hip) + calculate_distance(left_hip, left_knee) + calculate_distance(left_knee, left_ankle) + calculate_distance(left_ankle, left_heel)))
        print("ratio:" + str(ratio))
        
        if (len(buffer) > 20):
            buffer.pop(0)
        if (calculate_angle(left_hip, left_knee, left_ankle)>90 or calculate_angle(right_hip, right_knee, right_ankle)>90):
            buffer.append(ratio)
        else:
            print("腳不直")

        if np.median(buffer) > 0.10:
            # 利用連接阜傳送座標
            data_to_send = f"{np.multiply(mid_heel[0], frame.shape[1]).astype(int)},{np.multiply(mid_heel[1], frame.shape[1]).astype(int)},{z}\n"
            # ser.write(data_to_send.encode())
            print(data_to_send)
            print("media_ration:" + str(np.median(buffer)))
            print("小孩")
        else:
            print("media_ration:" + str(np.median(buffer)))
            print("大人")
            
        cv2.circle(frame, (np.multiply(mid_eye[0],frame.shape[1]).astype(int), np.multiply(mid_eye[1],frame.shape[0]).astype(int)), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.circle(frame, (np.multiply(mid_shoulder[0],frame.shape[1]).astype(int), np.multiply(mid_shoulder[1],frame.shape[0]).astype(int)), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.circle(frame, (np.multiply(mid_hip[0],frame.shape[1]).astype(int), np.multiply(mid_hip[1],frame.shape[0]).astype(int)), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.putText(frame, str(calculate_angle(left_hip, left_knee, left_ankle)), 
                        tuple(np.multiply(left_knee, [frame.shape[1], frame.shape[0]]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, 
                    )
        cv2.putText(frame, str(calculate_angle(right_hip, right_knee, right_ankle)), 
                        tuple(np.multiply(right_knee, [frame.shape[1], frame.shape[0]]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, 
                    )
        
    else:
        # 沒有檢測到關鍵點時的處理
        print("未檢測到姿勢")

    end_time = time.time()
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, "FPS: " + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 顯示結果影像
    cv2.imshow('MediaPipe_Pose_Detection', frame)
    # 按下"q"鍵退出程式
    if cv2.waitKey(1) == ord('q'):
        break
# 釋放資源
pose_detection.close()
cap.release()
cv2.destroyAllWindows()