import cv2
import os
import time
from face_detection import load_face_cascade, detect_faces

# --- Configuration ---
DATA_DIR = 'dataset'
CLASSES = ['attentive', 'distracted']
SETS = ['train', 'val']

# Number of images to capture for each class/set
NUM_IMAGES_PER_CLASS_TRAIN = 100
NUM_IMAGES_PER_CLASS_VAL = 30

def create_directories():
    """Creates the necessary directories for storing the dataset."""
    for s in SETS:
        for c in CLASSES:
            path = os.path.join(DATA_DIR, s, c)
            os.makedirs(path, exist_ok=True)
            print(f"Directory created: {path}")

def collect_data():
    """
    Uses the webcam to collect face images for training and validation.
    """
    face_cascade = load_face_cascade()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    for s in SETS:
        num_images = NUM_IMAGES_PER_CLASS_TRAIN if s == 'train' else NUM_IMAGES_PER_CLASS_VAL
        for c in CLASSES:
            print(f"\n--- Collecting images for class '{c}' in set '{s}' ---")
            print(f"Please look {c}. Press 's' to start capturing {num_images} images.")

            # Wait for user to get ready
            while True:
                ret, frame = cap.read()
                cv2.putText(frame, f"Set: {s}, Class: {c}. Press 's' to start.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow('Data Collector', frame)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break
            
            # Countdown before starting
            for i in range(3, 0, -1):
                ret, frame = cap.read()
                cv2.putText(frame, f"Starting in {i}...", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                cv2.imshow('Data Collector', frame)
                cv2.waitKey(1000)

            
            count = 0
            while count < num_images:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detect_faces(gray, face_cascade)

                if len(faces) > 0:
                    # Assume the largest face is the user
                    (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Save the cropped face
                    face_roi = gray[y:y+h, x:x+w]
                    img_name = f"{c}_{int(time.time() * 1000)}.jpg"
                    save_path = os.path.join(DATA_DIR, s, c, img_name)
                    cv2.imwrite(save_path, face_roi)
                    
                    count += 1
                    print(f"Saved {save_path} ({count}/{num_images})")

                # Display progress
                cv2.putText(frame, f"Capturing: {count}/{num_images}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Data Collector', frame)
                
                # Allow for a small delay and check for quit key
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

    print("\nData collection complete!")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    create_directories()
    collect_data()