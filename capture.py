import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import time

from model import AttentionCNN
from face_detection import detect_faces, load_face_cascade
from gaze_detection import detect_eyes, load_eye_cascade

# --- Constants and Model Loading ---

# Define the labels for classification
LABELS = {0: "Attentive", 1: "Distracted", 2: "Sleeping", 3: "Not Present"}
COLORS = {
    "Attentive": (0, 255, 0),   # Green
    "Distracted": (0, 255, 255), # Yellow
    "Sleeping": (0, 0, 255),     # Red
    "Not Present": (128, 128, 128) # Gray
}

# Load pre-trained models and cascades
face_cascade = load_face_cascade()
eye_cascade = load_eye_cascade()

# --- Model Loading ---
MODEL_PATH = 'attention_model.pth'
model = AttentionCNN(num_classes=2)

# Load the trained model if it exists, otherwise use heuristics
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    print("✅ Trained model loaded successfully.")
    USE_HEURISTICS = False
except FileNotFoundError:
    print("⚠️ Trained model not found. Falling back to eye-detection heuristics.")
    USE_HEURISTICS = True
model.eval()

# Image transformations for the model
preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def classify_attention(frame, state):
    """
    Processes a single frame to detect face, eyes, and classify attention.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, face_cascade)

    label = "Not Present"
    confidence = 1.0

    if len(faces) > 0:
        # Assume the largest face is the user
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        
        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        face_roi_gray = gray[y:y+h, x:x+w]
        
        # --- Sleeping Detection ---
        eyes = detect_eyes(face_roi_gray, eye_cascade)
        
        # Use time-based check for sleeping instead of frame count
        if len(eyes) == 0:
            if state['eyes_closed_start_time'] is None:
                state['eyes_closed_start_time'] = time.time()
        else:
            state['eyes_closed_start_time'] = None

        # If eyes have been closed for more than the threshold duration
        if state['eyes_closed_start_time'] and (time.time() - state['eyes_closed_start_time']) > state['eyes_closed_threshold_seconds']:
            label = "Sleeping"
            confidence = 1.0 # Heuristic-based, so confidence is 1.0
        else:
            # --- Hybrid Classification Logic ---
            # 1. Strong Heuristic: If at least one eye is visible, we assume Attentive.
            if len(eyes) >= 1:
                label = "Attentive"
                confidence = 1.0 # High confidence due to strong heuristic
            # 2. Fallback to CNN: If eye detection is ambiguous, trust the trained model.
            else:
                # If model is not available, use a simple heuristic as a last resort.
                if USE_HEURISTICS:
                    label = "Distracted"
                    confidence = 1.0
                else:
                    # Use the CNN for a more nuanced prediction.
                    pil_image = Image.fromarray(face_roi_gray)
                    image_tensor = preprocess(pil_image).unsqueeze(0)
                    with torch.no_grad():
                        outputs = model(image_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        label = LABELS[predicted.item()]
                        confidence = confidence.item()

    else: # No face detected
        state['eyes_closed_start_time'] = None # Reset timer if face is lost
        label = "Not Present"
        confidence = 1.0

    # --- Display the label on the frame ---
    status_color = COLORS.get(label, (255, 255, 255))
    
    # Create a black rectangle as a background for the text
    text_bg_y = y - 10 if len(faces) > 0 else 10
    cv2.rectangle(frame, (10, text_bg_y), (350, text_bg_y + 30), (0, 0, 0), -1)
    
    # Put the final status text on the frame
    cv2.putText(
        frame,
        f"Status: {label} ({confidence:.2f})",
        (15, text_bg_y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2
    )

    return frame, label, confidence, state