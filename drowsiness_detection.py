import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import subprocess

EYE_CLOSED_EAR_THRESH = 0.20
EYE_CLOSED_SECONDS = 2.5


ALARM_SOUND = "/System/Library/Sounds/Ping.aiff"


ALARM_INTERVAL = 0.3

LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

LEFT_EYE_POLY  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_POLY = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def eye_aspect_ratio(eye_pts):
    p1, p2, p3, p4, p5, p6 = eye_pts
    return (euclidean(p2, p6) + euclidean(p3, p5)) / (2.0 * euclidean(p1, p4) + 1e-6)


alarm_lock = threading.Lock()
alarm_proc = None
alarm_loop_on = False

def _play_sound():
    return subprocess.Popen(["afplay", ALARM_SOUND])

def _stop_proc():
    global alarm_proc
    if alarm_proc is not None and alarm_proc.poll() is None:
        try:
            alarm_proc.terminate()
        except:
            pass
    alarm_proc = None

def alarm_loop_worker():
    global alarm_proc, alarm_loop_on

    while True:
        with alarm_lock:
            if not alarm_loop_on:
                _stop_proc()
                return

            _stop_proc()
            alarm_proc = _play_sound()

        t0 = time.time()

        while time.time() - t0 < ALARM_INTERVAL:
            with alarm_lock:
                if not alarm_loop_on:
                    _stop_proc()
                    return
            time.sleep(0.01)

def start_alarm():
    global alarm_loop_on
    with alarm_lock:
        if alarm_loop_on:
            return
        alarm_loop_on = True

    threading.Thread(target=alarm_loop_worker, daemon=True).start()

def stop_alarm():
    global alarm_loop_on
    with alarm_lock:
        alarm_loop_on = False
        _stop_proc()


mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

closed_start = None
drowsy = False

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        status_text = "No Face"
        ear_value = None

        color_awake = (0,255,0)
        color_drowsy = (0,0,255)
        draw_color = (255,255,255)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            def lm_to_xy(idx):
                return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)

            left_eye = np.array([lm_to_xy(i) for i in LEFT_EYE_IDX])
            right_eye = np.array([lm_to_xy(i) for i in RIGHT_EYE_IDX])

            ear_value = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

            # Drowsiness detection
            if ear_value < EYE_CLOSED_EAR_THRESH:
                if closed_start is None:
                    closed_start = time.time()

                drowsy = (time.time() - closed_start) >= EYE_CLOSED_SECONDS
            else:
                closed_start = None
                drowsy = False

            if drowsy:
                start_alarm()
            else:
                stop_alarm()

            status_text = "DROWSY!" if drowsy else "Awake"
            draw_color = color_drowsy if drowsy else color_awake

            # Face box
            xs = np.array([p.x * w for p in lm])
            ys = np.array([p.y * h for p in lm])

            x1,x2 = int(xs.min()),int(xs.max())
            y1,y2 = int(ys.min()),int(ys.max())

            cv2.rectangle(frame,(x1,y1),(x2,y2),draw_color,2)

            # Eyes
            left_poly = np.array([lm_to_xy(i) for i in LEFT_EYE_POLY],dtype=np.int32)
            right_poly = np.array([lm_to_xy(i) for i in RIGHT_EYE_POLY],dtype=np.int32)

            cv2.polylines(frame,[left_poly],True,draw_color,2)
            cv2.polylines(frame,[right_poly],True,draw_color,2)

        else:
            stop_alarm()

        cv2.putText(frame,f"Status: {status_text}",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,draw_color,2)

        if ear_value is not None:
            cv2.putText(frame,f"EAR: {ear_value:.3f}",(20,80),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        cv2.imshow("Drowsiness Detection",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

stop_alarm()
cap.release()
cv2.destroyAllWindows()