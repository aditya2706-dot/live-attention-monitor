import cv2
import os

def load_eye_cascade():
    """
    Loads the Haar Cascade for eye detection from OpenCV's data files.
    """
    # Path to the cascade file
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')
    
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar cascade file not found at {cascade_path}")
        
    eye_cascade = cv2.CascadeClassifier(cascade_path)
    return eye_cascade

def detect_eyes(face_roi_gray, eye_cascade):
    """
    Detects eyes within a grayscale region of interest (the face).
    
    Args:
        face_roi_gray: The grayscale image of the detected face.
        eye_cascade: The loaded Haar Cascade classifier for eyes.
        
    Returns:
        A list of (x, y, w, h) tuples for each detected eye.
    """
    return eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))