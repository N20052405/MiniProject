import cv2
import mediapipe as mp
import numpy as np
import pickle
from flask import Flask, render_template, Response, redirect, url_for
from collections import deque
import sys
import os

# --- Flask App Setup ---
app = Flask(__name__, template_folder='.')

# --- Configuration & Global Model Loading ---
MODEL_PATH = 'sign_language_model_rf.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pickle'
PREDICTION_QUEUE_SIZE = 5 # Number of frames to average prediction over for smoothing

# Global resources
model = None
label_encoder = None
mp_hands = mp.solutions.hands
hands = None
cap = None
prediction_queue = deque(maxlen=PREDICTION_QUEUE_SIZE) # Queue for smoothing

def load_resources():
    """Loads the model, encoder, and initializes Mediapipe/Webcam."""
    global model, label_encoder, hands, cap
    try:
        # Load the trained model and label encoder
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Initialize Mediapipe
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize OpenCV Capture
        cap = cv2.VideoCapture(0)
        
        print("Model, Encoder, and Camera initialized successfully.")
        return True
    except FileNotFoundError:
        print(f"FATAL ERROR: Model files not found. Ensure you ran the model_trainer script.")
        return False
    except Exception as e:
        print(f"FATAL ERROR during resource loading: {e}")
        return False

# Load resources once when the application starts
if not load_resources():
    sys.exit(1)

mp_drawing = mp.solutions.drawing_utils

# --- Helper Function (Feature Extraction) ---
def extract_keypoints(results):
    """ Extracts and flattens the keypoints for 2 hands (126 features total). """
    left_hand = np.zeros(21*3)
    right_hand = np.zeros(21*3)
    
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            arr = np.array([[lmk.x, lmk.y, lmk.z] for lmk in hand_landmarks.landmark]).flatten()
            
            if handedness == 'Right':
                right_hand = arr
            elif handedness == 'Left':
                left_hand = arr
                
    return np.concatenate([left_hand, right_hand])

# --- Video Stream Generator ---
def generate_frames():
    """
    Captures frames, processes them for prediction, applies smoothing, 
    and yields JPEG frames.
    """
    global prediction_queue
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        # 1. Process frame with MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        current_prediction = "Waiting for hands..."

        # 2. Prediction Logic
        if results.multi_hand_landmarks:
            # Draw landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            keypoints = extract_keypoints(results)
            X_test_sample = keypoints.reshape(1, -1)
            
            try:
                # Get the prediction (encoded index)
                prediction_encoded = model.predict(X_test_sample)[0]
                
                # --- APPLY SMOOTHING ---
                prediction_queue.append(prediction_encoded)
                
                # Find the most frequent prediction in the queue (reduces flicker)
                most_common_prediction_encoded = max(set(prediction_queue), key=prediction_queue.count)
                
                # Convert the most common encoded index back to the sign label
                current_prediction = label_encoder.inverse_transform([most_common_prediction_encoded])[0]
                
            except Exception:
                current_prediction = "Processing Error"
        
        if not results.multi_hand_landmarks:
            current_prediction = "Waiting for hands..."
        
        # 3. Draw Prediction Text onto the frame
        cv2.putText(image, f'Prediction: {current_prediction}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 4. Encode the frame as JPEG (optimized fast compression)
        ret, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_bytes = buffer.tobytes()

        # 5. Yield (send) the frame to the browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Flask Routes ---

@app.route('/')
def index():
    """Default route, redirects to the sign-in page."""
    return redirect(url_for('signin'))

@app.route('/signin')
def signin():
    """Renders the Sign In Page."""
    return render_template('signin.html')

@app.route('/home')
def home():
    """Renders the Home page after successful (mock) sign-in."""
    return render_template('home.html')

@app.route('/detector')
def detector():
    """Renders the Real-Time Detector page."""
    return render_template('detector.html')

@app.route('/video_feed')
def video_feed():
    """This route streams the video frames."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Main Run ---
if __name__ == '__main__':
    print("\n--- Starting Sign Language Detector Web Server ---")
    print("Go to http://127.0.0.1:5000/ to access the application.")
    app.run(host='0.0.0.0', port=5000, debug=True)