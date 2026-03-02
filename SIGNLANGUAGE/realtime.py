import cv2
import mediapipe as mp
import numpy as np
import pickle
import sys

# --- Configuration ---
MODEL_PATH = 'sign_language_model_rf.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pickle'

# Load the trained model and label encoder
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
except FileNotFoundError:
    print("FATAL ERROR: Model or Label Encoder not found. Run the training script first.")
    sys.exit(1)

# MediaPipe Setup (Same as data collector)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- Helper Function (Reused from Phase 1) ---
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
                handedness = 'Right' # Default fallback
            
            arr = np.array([[lmk.x, lmk.y, lmk.z] for lmk in hand_landmarks.landmark]).flatten()
            
            if handedness == 'Right':
                right_hand = arr
            elif handedness == 'Left':
                left_hand = arr
                
    return np.concatenate([left_hand, right_hand])

# --- Real-Time Loop ---
cap = cv2.VideoCapture(0)
predicted_sign = "WAITING..."

print("Starting Real-Time Detection. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    # Process frame with MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 1. Feature Extraction & Drawing
    if results.multi_hand_landmarks:
        # Draw landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Extract 126 keypoints
        keypoints = extract_keypoints(results)
        
        # 2. Prediction
        X_test_sample = keypoints.reshape(1, -1)
        
        # Get prediction (encoded number)
        prediction_encoded = model.predict(X_test_sample)[0]
        
        # Convert prediction back to sign label
        predicted_sign = label_encoder.inverse_transform([prediction_encoded])[0]

    # 3. Display Result
    cv2.putText(image, f'Prediction: {predicted_sign}', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Sign Language Detector', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()