import cv2
import numpy as np
import time
from collections import Counter
from keras.models import load_model

# Load models
gender_model = load_model("gender_model.h5")
age_model = load_model("age_model.h5")

# Gender labels
gender_labels = ['Male', 'Female']

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

# Lists to store predictions
gender_votes = []
age_predictions = []

# Collect data for 10 seconds
start_time = time.time()
duration = 10  # seconds

print("🎥 Please look at the camera for 10 seconds...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
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

        # Predict age (denormalized)
        age_pred = age_model.predict(face_input, verbose=0)
        predicted_age = int(age_pred[0][0] * 100)
        age_predictions.append(predicted_age)

        # Optional: live rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # Show the webcam
    cv2.imshow("Collecting... Please Wait", frame)

    # Break after 10 seconds or on 'q'
    if time.time() - start_time > duration:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Show final result
if gender_votes and age_predictions:
    final_gender = gender_labels[Counter(gender_votes).most_common(1)[0][0]]
    final_age = int(np.mean(age_predictions))

    # Black canvas for display
    result_img = np.zeros((300, 600, 3), dtype=np.uint8)

    # ASCII version of result
    title = "== FINAL RESULT =="
    gender_text = f"> Gender: {final_gender}"
    age_text = f"> Age: {final_age} years"

    # Draw results
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
