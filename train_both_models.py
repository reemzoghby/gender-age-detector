import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("🔁 Loading dataset from UTKFace/crop_part1...")

DATASET_PATH = "UTKFace/crop_part1"
images = []
genders = []
ages = []

for root, _, files in os.walk(DATASET_PATH):
    for filename in files:
        if filename.endswith(".jpg"):
            try:
                parts = filename.split('_')
                age = int(parts[0])
                gender = int(parts[1])
                if age > 100 or gender not in [0, 1]:
                    continue
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    images.append(img)
                    genders.append(gender)
                    ages.append(age)
            except:
                continue

print(f"✅ Loaded {len(images)} images")

# Normalize
X = np.array(images) / 255.0
y_gender = to_categorical(np.array(genders), num_classes=2)
y_age = np.array(ages) / 100.0  # Normalize age [0, 1]

# Split
X_train, X_test, y_gender_train, y_gender_test = train_test_split(X, y_gender, test_size=0.2, random_state=42)
_, _, y_age_train, y_age_test = train_test_split(X, y_age, test_size=0.2, random_state=42)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4)

# === 🎓 Gender model ===
print("🎓 Training gender model...")

gender_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

gender_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

gender_model.fit(
    X_train, y_gender_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

gender_model.save("gender_model.h5")
print("✅ Saved gender_model.h5")

# === 🎓 Age model ===
print("🎓 Training age model...")

age_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1)  # Age regression output
])

age_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

age_model.fit(
    X_train, y_age_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

age_model.save("age_model.h5")
print("✅ Saved age_model.h5")
