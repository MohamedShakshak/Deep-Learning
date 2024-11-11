import cv2
import mediapipe as mp
import time
from joblib import load

# Initialize MediaPipe drawing and hands models
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands.Hands()

# Load the trained SVC model and MinMaxScaler
MLP = load("D://yolo8//best_model_5.joblib")
minmax = load('minmax.joblib')

# Label Mapping dictionary for encoding
Label_Mapping = {'a': 0, 'c': 1, 'eight': 2, 'f': 3, 'four': 4, 'g': 5, 'h': 6, 'help': 7,
                 'house': 8, 'i love you': 9, 'one': 10, 'please': 11, 'six': 12, 'three': 13,
                 'two': 14, 'w': 15, 'x': 16, 'yes': 17, 'z': 18}

# Reverse the Label_Mapping to get the prediction from integer
reverse_Label_Mapping = {v: k for k, v in Label_Mapping.items()}

# Set up video capture
cap = cv2.VideoCapture("D:\\yolo8\\help.mp4")
pTime = 0  # Previous time for calculating FPS
pos_estimation = []  # List to store position data

while True:
    ret, image = cap.read()
    if not ret:
        break

    # Get image dimensions
    rows, cols, _ = image.shape

    # Convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = mp_hands.process(image)
    image.flags.writeable = True

    # Process landmarks if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            
            # Extract x and y coordinates for each landmark and store in frame_data
            frame_data = []
            for landmark in hand_landmarks.landmark:
                x_coord = int(landmark.x * cols)
                y_coord = int(landmark.y * rows)
                frame_data.extend([x_coord, y_coord])  # Store both x and y for each landmark

            # Append frame data to pos_estimation
            pos_estimation.append(frame_data)

            # Convert pos_estimation to numpy array if needed, then apply scaling and prediction
            Pos_estimation_scaled = minmax.transform([frame_data])  # Scale the current frame data
            y_pred = MLP.predict(Pos_estimation_scaled)  # Make prediction for the current frame

            # Use the reverse Label Mapping to get the actual label from the predicted integer
            predicted_label = reverse_Label_Mapping[y_pred[0]]

            # Display prediction on the image
            cv2.putText(image, predicted_label, (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    # Convert image color back to BGR for display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, f'{fps:.0f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    # Show the output
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(10) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
