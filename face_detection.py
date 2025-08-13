import cv2
import numpy as np
import requests
import os

# Download required files automatically
def download_file(url, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {local_path}...")
        r = requests.get(url, stream=True)
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    else:
        print(f"{local_path} already exists")

# Download face detection model
FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
FACE_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_WEIGHTS_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

# Download emotion model (using Mini-Xception as alternative)
EMOTION_MODEL = "emotion_model.hdf5"
EMOTION_MODEL_URL = "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"

# Download all required files
download_file(FACE_MODEL_URL, FACE_PROTO)
download_file(FACE_WEIGHTS_URL, FACE_MODEL)
download_file(EMOTION_MODEL_URL, EMOTION_MODEL)

# Emotion labels
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Load models
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

# For emotion model we'll use Keras (install with: pip install tensorflow keras)
from keras.models import load_model
emotion_model = load_model(EMOTION_MODEL, compile=False)

def detect_emotions(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, 
                               (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Extract face ROI
            face = frame[startY:endY, startX:endX]
            try:
                # Preprocess face for emotion detection
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (64, 64))
                face = face.astype("float32") / 255.0
                face = np.expand_dims(face, axis=0)
                face = np.expand_dims(face, axis=-1)
                
                # Predict emotion
                preds = emotion_model.predict(face)[0]
                emotion = EMOTIONS[np.argmax(preds)]
                probability = np.max(preds)
                
                # Draw results
                label = f"{emotion}: {probability * 100:.1f}%"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing face: {e}")
    
    return frame

# Real-time detection
def realtime_detection():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = detect_emotions(frame)
        cv2.imshow("Emotion Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Image detection
def image_detection(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image")
        return
        
    result = detect_emotions(image)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("output.jpg", result)

if __name__ == "__main__":
    print("1. Real-time emotion detection")
    print("2. Detect emotions in image")
    choice = input("Select option (1/2): ")
    
    if choice == "1":
        realtime_detection()
    elif choice == "2":
        img_path = input("Enter image path: ")
        image_detection(img_path)
    else:
        print("Invalid choice")