import cv2
import mediapipe as mp
import math
import time
from collections import deque

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
NOSE_TIP = 1  # For head movement detection

# Parameters
EYE_AR_THRESH = 0.21
EYE_AR_CONSEC_FRAMES = 3
HEAD_MOVEMENT_THRESHOLD = 15  # Pixels
FATIGUE_THRESHOLD_SEC = 5

# Buffers and Counters
ear_history = deque(maxlen=5)
both_blink_count = 0
frame_counter = 0
eye_closed = False
fatigue_start_time = None
prev_nose_pos = None

# Helpers
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_ear(eye_landmarks):
    A = euclidean(eye_landmarks[1], eye_landmarks[5])
    B = euclidean(eye_landmarks[2], eye_landmarks[4])
    C = euclidean(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)

# Webcam Setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w = frame.shape[:2]

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Head movement check
        nose = landmarks[NOSE_TIP]
        nose_pos = (int(nose.x * w), int(nose.y * h))

        if prev_nose_pos:
            dx = abs(nose_pos[0] - prev_nose_pos[0])
            dy = abs(nose_pos[1] - prev_nose_pos[1])
            if dx > HEAD_MOVEMENT_THRESHOLD or dy > HEAD_MOVEMENT_THRESHOLD:
                prev_nose_pos = nose_pos
                cv2.putText(frame, "Head moved: skipping", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Blink Counter", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
        prev_nose_pos = nose_pos

        # Eyes
        left_eye = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in LEFT_EYE]
        right_eye = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in RIGHT_EYE]
        left_ear = get_ear(left_eye)
        right_ear = get_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Smooth EAR
        ear_history.append(avg_ear)
        smoothed_ear = sum(ear_history) / len(ear_history)

        # Blink detection (both eyes)
        if smoothed_ear < EYE_AR_THRESH:
            frame_counter += 1
            if frame_counter >= EYE_AR_CONSEC_FRAMES and not eye_closed:
                both_blink_count += 1
                eye_closed = True
        else:
            frame_counter = 0
            eye_closed = False

        # Fatigue detection
        if smoothed_ear < EYE_AR_THRESH:
            if fatigue_start_time is None:
                fatigue_start_time = time.time()
            elif time.time() - fatigue_start_time > FATIGUE_THRESHOLD_SEC:
                cv2.putText(frame, "💤 Fatigue Detected!", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            fatigue_start_time = None

        # Display counters
        cv2.putText(frame, f"Blink Count: {both_blink_count}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Optional: Eye visualization
        for pt in left_eye + right_eye:
            cv2.circle(frame, pt, 2, (100, 255, 255), -1)

    cv2.imshow("Blink Counter", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
