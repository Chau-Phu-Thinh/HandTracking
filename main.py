import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"

# Point-pair connections for the 21 hand landmarks (replaces the old mp_hands.HAND_CONNECTIONS)
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index finger
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle finger
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring finger
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky finger
    (0, 17),  # wrist - pinky base
]

# BGR colors (OpenCV uses BGR, not RGB)
POINT_COLOR = (255, 255, 0)  # cyan
LINE_COLOR = (255, 255, 255)  # white


def draw_landmarks(frame, hand_landmarks_list, handedness_list):
    """Manually draw landmarks + connections with OpenCV (replaces the old mp_drawing.draw_landmarks)."""
    h, w, _ = frame.shape
    for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        # draw connecting lines
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], LINE_COLOR, 2)

        # draw joint points
        for x, y in points:
            cv2.circle(frame, (x, y), 4, POINT_COLOR, -1)

        # Left/Right label
        label = handedness[0].category_name
        cv2.putText(
            frame, label, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
        )


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"'{MODEL_PATH}' not found. Download the model from: "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task"
        )

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame!")
            break

        # frame = cv2.flip(frame, 1)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # timestamp must strictly increase in milliseconds for VIDEO mode
        frame_timestamp_ms = int((time.time() - start_time) * 1000)

        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if result.hand_landmarks:
            draw_landmarks(frame, result.hand_landmarks, result.handedness)

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
