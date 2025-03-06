import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load model
model_path = "face_model.h5"
try:
    model = load_model(model_path)
except Exception as e:
    print("Error loading model:", str(e))
    exit()

# Emotion Labels
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start Webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect Faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(30,30))

        for (x, y, w, h) in faces:
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Preprocess Face for Model
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48,48))
            face_roi = np.array(face_roi, dtype=np.float32) / 255.0
            face_roi = np.expand_dims(face_roi, axis=-1)
            face_roi = np.expand_dims(face_roi, axis=0)

            # Predict Emotion
            predictions = model.predict(face_roi)
            emotion_label = class_labels[np.argmax(predictions)]
            emotion_confidence = np.max(predictions)

            # Display Emotion
            text = f'{emotion_label} ({emotion_confidence:.2f})'
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Show Output
        cv2.imshow('Emotion Detection', frame)

        # Break on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
