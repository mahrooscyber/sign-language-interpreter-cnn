
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

IMAGE_SIZE = 50
MODEL_PATH = "cnn_model_keras2.h5"

def load_pickle(name):
    with open(name, "rb") as f:
        return np.asarray(pickle.load(f))

X_train = load_pickle("train_images")
y_train = load_pickle("train_labels")
X_val = load_pickle("val_images")
y_val = load_pickle("val_labels")
X_test = load_pickle("test_images")
y_test = load_pickle("test_labels")

num_classes = len(set(y_train.tolist()) | set(y_val.tolist()) | set(y_test.tolist()))
if num_classes < 2:
    raise ValueError("Need at least two gesture classes to train this classifier.")

def prepare_images(X):
    X = X.astype("float32") / 255.0
    return X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

X_train = prepare_images(X_train)
X_val = prepare_images(X_val)
X_test = prepare_images(X_test)

model = models.Sequential([
    layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

save_best = callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
early_stop = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=6,
    restore_best_weights=True
)

model.summary()
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[save_best, early_stop]
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {accuracy * 100:.2f}%")

model.save(MODEL_PATH)
print("Saved model:", MODEL_PATH)
