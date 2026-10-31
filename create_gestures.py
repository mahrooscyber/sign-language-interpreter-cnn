
import cv2
import numpy as np
import pickle
import os
import sqlite3
import random

IMAGE_X, IMAGE_Y = 50, 50
TOTAL_PICS = 150


def get_hand_hist():
    with open("hist", "rb") as f:
        return pickle.load(f)


def init_create_folder_database():
    os.makedirs("gestures", exist_ok=True)

    with sqlite3.connect("gesture_db.db") as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gesture (
                g_id INTEGER PRIMARY KEY AUTOINCREMENT,
                g_name TEXT NOT NULL
            )
        """)


def create_folder(folder_name):
    os.makedirs(folder_name, exist_ok=True)


def store_in_db(g_id, g_name):
    with sqlite3.connect("gesture_db.db") as conn:
        existing = conn.execute(
            "SELECT g_name FROM gesture WHERE g_id = ?", (g_id,)
        ).fetchone()

        if existing:
            print(f"Gesture ID {g_id} already exists: {existing[0]}")
            choice = input("Update its name? (y/n): ").strip().lower()
            if choice == "y":
                conn.execute(
                    "UPDATE gesture SET g_name = ? WHERE g_id = ?",
                    (g_name, g_id)
                )
                return True
            print("Cancelled. Choose another gesture ID.")
            return False

        conn.execute(
            "INSERT INTO gesture (g_id, g_name) VALUES (?, ?)",
            (g_id, g_name)
        )
        return True


def open_camera():
    for index in (0, 1):
        cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cam.isOpened():
            ok, frame = cam.read()
            if ok and frame is not None:
                print(f"Using camera index {index}")
                return cam
        cam.release()

    return None


def store_images(g_id):
    hist = get_hand_hist()
    cam = open_camera()

    if cam is None:
        print("ERROR: Could not open a working webcam.")
        return

    x, y, w, h = 300, 100, 300, 300
    folder = os.path.join("gestures", str(g_id))
    create_folder(folder)

    # Continue numbering if this folder already has images.
    existing = [
        int(os.path.splitext(name)[0])
        for name in os.listdir(folder)
        if name.lower().endswith(".jpg")
        and os.path.splitext(name)[0].isdigit()
    ]
    pic_no = max(existing, default=0)

    if pic_no >= TOTAL_PICS:
        print(f"This folder already has {pic_no} images.")
        cam.release()
        return

    capturing = False
    frames = 0
    print("Place your hand inside the green box.")
    print("Press C to start/pause capture. Press Q to quit.")

    try:
        while pic_no < TOTAL_PICS:
            ok, img = cam.read()
            if not ok or img is None:
                print("ERROR: Webcam frame unavailable.")
                break

            img = cv2.flip(img, 1)
            height, width = img.shape[:2]

            # Keep the region of interest inside the camera frame.
            x1_box = min(x, width - 1)
            y1_box = min(y, height - 1)
            x2_box = min(x + w, width)
            y2_box = min(y + h, height)

            if x2_box <= x1_box or y2_box <= y1_box:
                print("Camera resolution is too small for the capture box.")
                break

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject(
                [hsv], [0, 1], hist, [0, 180, 0, 256], 1
            )

            disc = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (10, 10)
            )
            cv2.filter2D(dst, -1, disc, dst)
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)

            thresh_full = cv2.threshold(
                blur, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]

            thresh = thresh_full[y1_box:y2_box, x1_box:x2_box]
            contours_result = cv2.findContours(
                thresh.copy(),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )
            contours = contours_result[-2]

            valid_contours = []
            for contour in contours:
                if contour is None or contour.size == 0:
                    continue

                contour = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
                if len(contour) >= 3:
                    valid_contours.append(contour)

            if capturing:
                frames += 1

            if valid_contours:
                contour = max(valid_contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)

                if capturing and area > 10000 and frames > 50:
                    bx, by, bw, bh = cv2.boundingRect(contour)
                    save_img = thresh[by:by + bh, bx:bx + bw]

                    if save_img.size > 0:
                        side = max(bw, bh)
                        top = (side - bh) // 2
                        bottom = side - bh - top
                        left = (side - bw) // 2
                        right = side - bw - left

                        save_img = cv2.copyMakeBorder(
                            save_img, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=0
                        )
                        save_img = cv2.resize(
                            save_img, (IMAGE_X, IMAGE_Y)
                        )

                        if random.randint(0, 1) == 0:
                            save_img = cv2.flip(save_img, 1)

                        next_no = pic_no + 1
                        image_path = os.path.join(
                            folder, f"{next_no}.jpg"
                        )
                        if cv2.imwrite(image_path, save_img):
                            pic_no = next_no

            cv2.rectangle(
                img, (x1_box, y1_box), (x2_box, y2_box),
                (0, 255, 0), 2
            )
            status = "CAPTURING" if capturing else "PAUSED - press C"
            cv2.putText(
                img, f"{status}  Images: {pic_no}/{TOTAL_PICS}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2
            )

            cv2.imshow("Capturing gesture", img)
            cv2.imshow("Hand threshold", thresh)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("c"):
                capturing = not capturing
                if capturing:
                    frames = 0
                    print("Capture started.")
                else:
                    print("Capture paused.")

    finally:
        cam.release()
        cv2.destroyAllWindows()

    print(f"Saved {pic_no} images in {folder}")


if __name__ == "__main__":
    init_create_folder_database()

    raw_id = input("Enter numeric gesture ID (e.g. 1): ").strip()
    if not raw_id.isdigit():
        print("ERROR: Gesture ID must be a positive integer.")
        raise SystemExit(1)

    g_id = int(raw_id)
    if g_id < 1:
        print("ERROR: Use a gesture ID of 1 or greater.")
        raise SystemExit(1)

    g_name = input("Enter gesture name/text (e.g. Namaste): ").strip()
    if not g_name:
        print("ERROR: Gesture name cannot be empty.")
        raise SystemExit(1)

    if store_in_db(g_id, g_name):
        store_images(g_id)
