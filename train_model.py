import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

# Path to your actual dataset folder (deep inside the subfolders)
DATASET_PATH = "UTKFace/utkface_aligned_cropped/crop_part1/"

# Data lists
images = []
genders = []
ages = []

# Recursively load images from all subfolders
for root, dirs, files in os.walk(DATASET_PATH):
    for filename in files:
        if filename.endswith(".jpg"):
            try:
                parts = filename.split('_')
                if len(parts) < 2:
                    continue

                age = int(parts[0])
                gender = int(parts[1])

                # ❗ Filter out unexpected gender values
                if gender not in [0, 1]:
                    continue

                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                img = cv2.resize(img, (64, 64))
                images.append(img)
                genders.append(gender)
                ages.append(age)

            except Exception as e:
                print(f"Skipping {filename}: {e}")
                continue

print(f"✅ Loaded {len(images)} valid images.")

# Convert data to numpy
X = np.array(images)
y_gender = to_categorical(genders, num_classes=2)
X = X / 255.0  # Normalize pixel values

# Split dataset
X_train, X_test, y_gender_train, y_gender_test = train_test_split(
    X, y_gender, test_size=0.2, random_state=42
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 outputs for gender: male / female
])

# Compile model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_gender_train, epochs=10, batch_size=32, validation_split=0.1)

# Save trained model
model.save("gender_model.h5")
print("✅ Model saved as gender_model.h5")
