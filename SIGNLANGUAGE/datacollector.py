import cv2
import mediapipe as mp
import numpy as np
import pickle
import sys

# --- Configuration ---
DATA_PATH = 'sign_language_data.pickle' # Renamed for clarity
# List of signs (Gestures: 20+ from the Home Page spec)
ACTIONS = [
    'hello', 'thankyou', 'yes', 'no', 'bad', 'good', 'okay', 'done', 
    'sorry', 'please', 'help', 'college', 'teach', 'we', 'you', 
    'family', 'home', 'come', 'go', 'begin', 'stop', 'city', 'goodluck','where' ,'book'
]
NO_OF_SAMPLES = 30 # Number of data points (frames) to collect per sign

# MediaPipe Setup for Hand Landmarks (up to 2 hands)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

data = {} # Dictionary to hold all collected data

# --- Helper Function for Feature Extraction ---
def extract_keypoints(results):
    """ Extracts and flattens the keypoints for 2 hands (126 features total). """
    left_hand = np.zeros(21*3)
    right_hand = np.zeros(21*3)
    
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Check if handedness list is available before accessing
            if results.multi_handedness and len(results.multi_handedness) > idx:
                handedness = results.multi_handedness[idx].classification[0].label
            else:
                # Default to Right hand if handedness info is missing (basic fallback)
                handedness = 'Right' 
            
            arr = np.array([[lmk.x, lmk.y, lmk.z] for lmk in hand_landmarks.landmark]).flatten()
            
            # MediaPipe typically reports the hand relative to the user's camera view.
            # We map the landmarks to a consistent feature vector (Left Hand then Right Hand).
            if handedness == 'Right':
                right_hand = arr
            elif handedness == 'Left':
                left_hand = arr
                
    return np.concatenate([left_hand, right_hand])


# --- Main Data Collection Loop ---
cap = cv2.VideoCapture(0)

print("Starting data collection. Press 's' to start collecting a sign.")

for action in ACTIONS:
    print(f"\n--- Ready to collect data for: *{action}* ---")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'Sign: {action} (Press S to start)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('OpenCV Feed', frame)
        if cv2.waitKey(10) & 0xFF == ord('s'):
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    samples_collected = 0
    data[action] = []

    for sample in range(NO_OF_SAMPLES):
        ret, frame = cap.read()
        if not ret: break
            
        frame = cv2.flip(frame, 1)
        
        # Process frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks and extract features
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            keypoints = extract_keypoints(results)
            data[action].append(keypoints)
            samples_collected += 1

        # Display status
        cv2.putText(image, f'Collecting: {action} | Sample: {samples_collected}/{NO_OF_SAMPLES}', 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('OpenCV Feed', image)

        if samples_collected >= NO_OF_SAMPLES:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources and save data
cap.release()
cv2.destroyAllWindows()

if data:
    with open(DATA_PATH, 'wb') as f:
        pickle.dump(data, f)
    print(f"\n--- Data collection complete. Saved {len(data)} signs to {DATA_PATH} ---")
else:
    print("\n--- No data was collected. Ensure your camera works and hands are visible. ---")