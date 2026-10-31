
import cv2
import numpy as np
import pickle
import sqlite3
from tensorflow.keras.models import load_model

MODEL_PATH = "cnn_model_keras2.h5"
HIST_PATH = "hist"
DB_PATH = "gesture_db.db"
IMAGE_SIZE = 50

# Load the trained model
model = load_model(MODEL_PATH)

# Load the saved hand histogram
with open(HIST_PATH, "rb") as f:
    hist = pickle.load(f)

# Load gesture names from the database
conn = sqlite3.connect(DB_PATH)
rows = conn.execute(
    "SELECT g_id, g_name FROM gesture ORDER BY g_id"
).fetchall()
conn.close()

# Training folders 1-5 correspond to model classes 0-4
gesture_names = {int(g_id) - 1: name for g_id, name in rows}
print("Loaded gestures:", gesture_names)

# Open webcam
cap = None
for camera_index in (0, 1):
    test_cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if test_cap.isOpened():
        ok, frame = test_cap.read()
        if ok and frame is not None:
            cap = test_cap
            break
        test_cap.release()

if cap is None:
    raise RuntimeError("Could not open webcam.")

print("Webcam started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]

    # Fixed hand region, similar to the collection workflow
    x1, y1 = int(width * 0.55), int(height * 0.15)
    x2, y2 = int(width * 0.95), int(height * 0.85)

    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Segment skin using the saved histogram
    back_projection = cv2.calcBackProject(
        [hsv], [0, 1], hist, [0, 180, 0, 256], 1
    )

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.filter2D(back_projection, -1, kernel)
    _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

    mask = cv2.medianBlur(mask, 5)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find the largest valid contour
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    prediction_text = "Show hand in box"

    valid_contours = [
        c for c in contours
        if cv2.contourArea(c) > 1000
    ]

    if valid_contours:
        contour = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)

        hand = mask[y:y + h, x:x + w]

        # Preserve aspect ratio and center on a square canvas
        side = max(w, h)
        square = np.zeros((side, side), dtype=np.uint8)

        offset_x = (side - w) // 2
        offset_y = (side - h) // 2
        square[offset_y:offset_y + h, offset_x:offset_x + w] = hand

        processed = cv2.resize(
            square, (IMAGE_SIZE, IMAGE_SIZE)
        )

        model_input = processed.astype("float32") / 255.0
        model_input = model_input.reshape(
            1, IMAGE_SIZE, IMAGE_SIZE, 1
        )

        probabilities = model.predict(model_input, verbose=0)[0]
        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])

        if confidence >= 0.70:
            gesture_name = gesture_names.get(
                class_id, "Unknown"
            )
            prediction_text = (
                f"{gesture_name} ({confidence * 100:.1f}%)"
            )
        else:
            prediction_text = "Uncertain"

        cv2.rectangle(
            frame,
            (x1 + x, y1 + y),
            (x1 + x + w, y1 + y + h),
            (0, 255, 0),
            2
        )

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    cv2.putText(
        frame,
        prediction_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Gesture Recognition", frame)
    cv2.imshow("Hand Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
