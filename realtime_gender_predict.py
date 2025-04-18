import cv2
import numpy as np
from keras.models import load_model

# Load your trained model
model = load_model("gender_model.h5")

# Labels: 0 = Male, 1 = Female
labels = ['Male', 'Female']

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

print("📸 Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through each detected face
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (64, 64))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        # Predict gender
        prediction = model.predict(face_img)
        gender_index = np.argmax(prediction)
        gender = labels[gender_index]
        confidence = round(np.max(prediction) * 100, 1)

        # Draw box and label
        label_text = f"{gender} ({confidence}%)"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    # Show the video
    cv2.imshow('Real-Time Gender Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
