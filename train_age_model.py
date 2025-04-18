import os
import cv2
import numpy as np
from keras.models import load_model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from collections import Counter
import time

# Load gender model
gender_model = load_model("gender_model.h5")

# Check if age model exists
if not os.path.exists("age_model.h5"):
    print("🔄 age_model.h5 not found... Training a new age model!")

    # Path to UTKFace images
    DATASET_PATH = "UTKFace/crop_part1"

    images = []
    ages = []

    for root, dirs, files in os.walk(DATASET_PATH):
        for filename in files:
            if filename.endswith(".jpg"):
                try:
                    age = int(filename.split('_')[0])
                    img_path = os.path.join(root, filename)
                    img = cv2.imread(img_path)

                    if img is not None:
                        img = cv2.resize(img, (64, 64))
                        images.append(img)
                        ages.append(age)
                except:
                    continue

    print(f"✅ Loaded {len(images)} age-labeled images.")

    # Prepare data
    X = np.array(images) / 255.0
    y = np.array(ages) / 100.0  # normalize age to [0, 1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build age model
    age_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1)  # Regression
    ])

    age_model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

    # Train model
    age_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
    age_model.save("age_model.h5")
    print("✅ New age model trained and saved as age_model.h5.")
else:
    age_model = load_model("age_model.h5")

# Labels
gender_labels = ['Male', 'Female']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
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
        predicted_age = int(age_pred[0][0] * 100)  # denormalize
        age_predictions.append(predicted_age)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow("Analyzing... Please Wait", frame)

    if time.time() - start_time > duration:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Final result
if gender_votes and age_predictions:
    final_gender = gender_labels[Counter(gender_votes).most_common(1)[0][0]]
    final_age = int(np.mean(age_predictions))

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
