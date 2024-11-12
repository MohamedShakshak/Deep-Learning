import numpy as np
import cv2
import mediapipe as mp
import time
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands.Hands()

cap = cv2.VideoCapture("D:\\yolo8\\z.mp4")
pTime = 0
pos_estimation = []

while True:
    ret, image = cap.read()
    if not ret:
        break

    rows, cols, _ = image.shape
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = mp_hands.process(image)
    image.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            
            # Extract the x and y coordinates for each landmark
            frame_data = []
            for landmark in hand_landmarks.landmark:
                x_coord = int(landmark.x * cols)
                y_coord = int(landmark.y * rows)
                frame_data.extend([x_coord, y_coord])  # Store both x and y for each landmark

            pos_estimation.append(frame_data)  # Append all coordinates for the frame

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, f'{fps:.0f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Convert to DataFrame and label
Pos_estimation_df = pd.DataFrame(pos_estimation)
Pos_estimation_df['label'] = "z"  # Add label column
Pos_estimation_df.to_csv("z.csv", index=False)
