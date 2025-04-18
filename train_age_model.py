import cv2
import numpy as np
from keras.models import load_model
from collections import Counter
import time

# Load trained models
gender_model = load_model("gender_model.h5")
age_model = load_model("age_model.h5")

# Labels
gender_labels = ['Male', 'Female']

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

gender_votes = []
age_predictions = []

start_time = time.time()
duration = 5  # seconds to scan

print("🎥 Please look at the camera for 5 seconds...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (64, 64))
        face_input = face_resized / 255.0
        face_input = np.expand_dims(face_input, axis=0)

        # Predict gender
        gender_pred = gender_model.predict(face_input, verbose=0)
        gender_index = int(np.argmax(gender_pred))
        gender_votes.append(gender_index)

        # Predict age
        age_pred = age_model.predict(face_input, verbose=0)
        predicted_age = int(age_pred[0][0] * 100)  # denormalized
        age_predictions.append(predicted_age)

        # Optional: draw live box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow("Analyzing... Please Wait", frame)

    if time.time() - start_time > duration:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Final result screen
if gender_votes and age_predictions:
    final_gender = gender_labels[Counter(gender_votes).most_common(1)[0][0]]
    final_age = int(np.mean(age_predictions))

    # --- Pretty final display ---
    result_frame = np.full((400, 700, 3), (30, 30, 30), dtype=np.uint8)

    title = "🎯 Analysis Result"
    gender_text = f"👤 Gender: {final_gender}"
    age_text = f"🎂 Estimated Age: {final_age} years"

    cv2.putText(result_frame, title, (90, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(result_frame, gender_text, (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(result_frame, age_text, (100, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    cv2.imshow("✅ FINAL RESULT", result_frame)
    print("✅ Final Gender:", final_gender)
    print("✅ Final Age:", final_age)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("⚠️ No face was detected.")
