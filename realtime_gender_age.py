# realtime_gender_age.py

import cv2
import numpy as np
import time
from collections import Counter
from keras.models import load_model

# Load models
gender_model = load_model("gender_model.h5")
age_model = load_model("age_model.h5")

gender_labels = ['Male', 'Female']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

gender_votes = []
age_predictions = []

start_time = time.time()
duration = 10  # seconds

print("🎥 Please look at the camera for 10 seconds...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        if w < 60 or h < 60:
            continue
        face_img = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (64, 64))
        face_input = face_resized / 255.0
        face_input = np.expand_dims(face_input, axis=0)

        gender_pred = gender_model.predict(face_input, verbose=0)
        gender_index = int(np.argmax(gender_pred))
        gender_votes.append(gender_index)

        age_pred = age_model.predict(face_input, verbose=0)
        predicted_age = int(age_pred[0][0] * 100)
        predicted_age = max(1, min(predicted_age, 100))
        age_predictions.append(predicted_age)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow("Analyzing... Please Wait", frame)

    if time.time() - start_time > duration:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if gender_votes and age_predictions:
    final_gender = gender_labels[Counter(gender_votes).most_common(1)[0][0]]
    final_age = int(np.median(age_predictions))

    result_img = np.zeros((300, 600, 3), dtype=np.uint8)
    title = "🎯 FINAL RESULT"
    gender_text = f"👤 Gender: {final_gender}"
    age_text = f"🎂 Age: {final_age} years"

    cv2.putText(result_img, title, (120, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(result_img, gender_text, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    cv2.putText(result_img, age_text, (100, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

    print("\n✅ Final Prediction:")
    print(f"Gender: {final_gender}")
    print(f"Age: {final_age} years")

    cv2.imshow("✅ FINAL RESULT", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("⚠️ No face detected.")
