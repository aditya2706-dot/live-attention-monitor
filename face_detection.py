import cv2
import os

def load_face_cascade():
    """
    Loads the Haar Cascade for face detection from OpenCV's data files.
    """
    # Path to the cascade file
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar cascade file not found at {cascade_path}")
        
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade

def detect_faces(gray_image, face_cascade):
    """
    Detects faces in a grayscale image.
    
    Args:
        gray_image: The grayscale image frame.
        face_cascade: The loaded Haar Cascade classifier for faces.
        
    Returns:
        A list of (x, y, w, h) tuples for each detected face.
    """
    return face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))