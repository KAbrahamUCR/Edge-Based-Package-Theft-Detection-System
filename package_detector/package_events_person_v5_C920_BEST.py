#!/usr/bin/env python3
"""
Package + Person Event Detector (DELIVERED / REMOVED)
Fixes false positives + flicker using:
- stricter conf/IoU
- min box area filter
- ROI (region of interest) filter
- rolling-window smoothing of package count
"""

import time
import cv2
from collections import deque, Counter
from ultralytics import YOLO

import json
import paho.mqtt.client as mqtt

import torch
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

# ---------------- CONFIG ----------------
PACKAGE_MODEL_PATH = "/home/nicholas/models/package_best.pt"
PERSON_MODEL_NAME = "yolov8n.pt"   # COCO pretrained

CAMERA_INDEX = 0

# Make detections stricter (reduces false positives)
PKG_CONF = 0.45
PKG_IOU  = 0.60

PERSON_CONF = 0.55
PERSON_IOU  = 0.50

# Faster person detection + less event flicker
PERSON_IMG_SIZE = 320
PERSON_HOLD_SECONDS = 1.0
IMG_SIZE = 1280

# Run inference at this rate
POLL_HZ = 6

# ---- Filters to kill false positives ----
# Ignore tiny boxes (fraction of full image area)
MIN_BOX_AREA_FRAC = 0.0010

# Only count packages inside this ROI (fractions of image)
ROI = (0.0, 0.0, 1.0, 1.0)

# ---- Flicker control / event debounce ----
STABLE_SECONDS = 0.8
DELIVERED_COOLDOWN_SECONDS = 2.0
REMOVED_COOLDOWN_SECONDS = 2.0
REMOVED_DELIVERED_COOLDOWN_SECONDS = 0.5
PERSON_STABLE_FRAMES = 2

# Delivery handling
ALLOW_DELIVERED_WITHOUT_PERSON = True
DELIVERED_STABLE_SECONDS_WITH_PERSON = 0.8
DELIVERED_STABLE_SECONDS_NO_PERSON = 3.0

# Removal handling
ALLOW_REMOVED_WITHOUT_PERSON = True
REMOVED_STABLE_SECONDS_WITH_PERSON = 0.8
REMOVED_STABLE_SECONDS_NO_PERSON = 3.0
REMOVED_PERSON_GRACE_SECONDS = 2.0

# Rolling window smoothing
SMOOTH_WINDOW = 3
# ----------------------------------------

SHOW_WINDOW = True
DRAW_BOXES  = True

# ------------- MQTT SETUP (Edge --> Fog) -------------
MQTT_HOST = "192.168.4.34"   # <-- your laptop IP
MQTT_PORT = 1883
MQTT_TOPIC = "Home/package_events"

mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)
mqtt_client.loop_start()

def send_event(event_type: str, pkg_count: int, person: bool):
    payload = {
        "ts": time.time(),
        "event": event_type,
        "pkg_count": pkg_count,
        "person": bool(person),
        "device": "jetson-orin",
    }
    mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)


def now() -> float:
    return time.time()


def has_person(coco_results) -> bool:
    if coco_results.boxes is None or len(coco_results.boxes) == 0:
        return False
    cls_ids = coco_results.boxes.cls
    return bool((cls_ids == 0).any().item())


def mode_int(values):
    c = Counter(values)
    return c.most_common(1)[0][0]


def count_filtered_packages(pkg_res, frame_shape):
    """Count packages after applying ROI + min-area filters."""
    H, W = frame_shape[:2]
    min_area = MIN_BOX_AREA_FRAC * (W * H)

    rx1, ry1, rx2, ry2 = ROI
    X1, Y1, X2, Y2 = int(rx1 * W), int(ry1 * H), int(rx2 * W), int(ry2 * H)

    if pkg_res.boxes is None or len(pkg_res.boxes) == 0:
        return 0

    xyxy = pkg_res.boxes.xyxy.cpu().numpy()
    confs = pkg_res.boxes.conf.cpu().numpy()

    kept = 0
    for (x1, y1, x2, y2), c in zip(xyxy, confs):
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area < min_area:
            continue

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        if not (X1 <= cx <= X2 and Y1 <= cy <= Y2):
            continue

        kept += 1

    return kept


def main():
    pkg_model = YOLO(PACKAGE_MODEL_PATH)
    person_model = YOLO(PERSON_MODEL_NAME)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Cam:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {CAMERA_INDEX}. Try 0 or 1.")

    infer_period = 1.0 / max(POLL_HZ, 1)
    last_infer = 0.0

    fps = 0
    frame_counter = 0
    fps_timer = now()

    confirmed_pkg_count = 0
    stable_candidate_count = None
    candidate_since = None
    last_delivered_time = 0.0
    last_removed_time = 0.0

    person_streak = 0
    person_present_stable = False
    person_last_seen = 0.0

    recent_counts = deque(maxlen=SMOOTH_WINDOW)

    print("Running. Press 'q' to quit.")
    print(f"PKG_CONF={PKG_CONF}, PKG_IOU={PKG_IOU}, MIN_BOX_AREA_FRAC={MIN_BOX_AREA_FRAC}, ROI={ROI}")
    print("DELIVERED can trigger with a person (fast) or without a person (slower).")
    print("REMOVED can trigger with a person (fast) or without a person (slower) if enabled.")

    while True:
        ok, frame = cap.read()

        frame_counter += 1
        if (now() - fps_timer) >= 1.0:
            fps = frame_counter
            frame_counter = 0
            fps_timer = now()
            print(f"FPS: {fps}", end="\r")

        if not ok:
            print("No frame from camera.")
            break

        t = now()

        if (t - last_infer) >= infer_period:
            last_infer = t

            infer_start = time.time()

            # --- Package detections ---
            pkg_res = pkg_model.predict(
                frame, conf=PKG_CONF, iou=PKG_IOU, imgsz=IMG_SIZE, device=0, verbose=False
            )[0]
            pkg_t = time.time()

            raw_pkg_count = count_filtered_packages(pkg_res, frame.shape)
            recent_counts.append(raw_pkg_count)
            smooth_pkg_count = mode_int(recent_counts) if len(recent_counts) > 0 else raw_pkg_count

            # --- Person detections ---
            person_res = person_model.predict(
                frame, conf=PERSON_CONF, iou=PERSON_IOU, imgsz=PERSON_IMG_SIZE, device=0, verbose=False
            )[0]
            infer_end = time.time()

            print(
                f"PKG: {(pkg_t - infer_start)*1000:.1f} ms | "
                f"PERSON: {(infer_end - pkg_t)*1000:.1f} ms | "
                f"TOTAL: {(infer_end - infer_start)*1000:.1f} ms"
            )

            person_now = has_person(person_res)

            if person_now:
                person_streak += 1
                person_last_seen = t
            else:
                person_streak = 0

            person_present_stable = (
                (person_streak >= PERSON_STABLE_FRAMES) or
                ((t - person_last_seen) <= PERSON_HOLD_SECONDS)
            )

            # --- Draw output ---
            if SHOW_WINDOW and DRAW_BOXES:
                frame_vis = pkg_res.plot()

                H, W = frame.shape[:2]
                rx1, ry1, rx2, ry2 = ROI
                X1, Y1, X2, Y2 = int(rx1 * W), int(ry1 * H), int(rx2 * W), int(ry2 * H)
                cv2.rectangle(frame_vis, (X1, Y1), (X2, Y2), (255, 255, 0), 2)

                cv2.putText(
                    frame_vis,
                    f"FPS: {fps} | PERSON: {'YES' if person_present_stable else 'NO'} | PKG: {raw_pkg_count}/{smooth_pkg_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0) if person_present_stable else (0, 0, 255),
                    2,
                )
            else:
                frame_vis = frame

            # --- Event logic ---
            in_delivered_cooldown = (t - last_delivered_time) < DELIVERED_COOLDOWN_SECONDS
            in_removed_cooldown   = (t - last_removed_time) < REMOVED_COOLDOWN_SECONDS

            if smooth_pkg_count != confirmed_pkg_count:
                if stable_candidate_count != smooth_pkg_count:
                    stable_candidate_count = smooth_pkg_count
                    candidate_since = t
                else:
                    if candidate_since is not None:
                        old = confirmed_pkg_count
                        new = stable_candidate_count
                        elapsed = (t - candidate_since)

                        if new > old:
                            # DELIVERED: allow two paths:
                            #  - fast path when a person is present
                            #  - slower path when no person is seen
                            allow_no_person_delivery = ALLOW_DELIVERED_WITHOUT_PERSON

                            if not in_delivered_cooldown:
                                if person_present_stable:
                                    if elapsed >= DELIVERED_STABLE_SECONDS_WITH_PERSON:
                                        confirmed_pkg_count = new
                                        print(f"[{time.strftime('%H:%M:%S')}] DELIVERED  (count {old} -> {confirmed_pkg_count})")
                                        send_event("DELIVERED", confirmed_pkg_count, person_present_stable)
                                        last_delivered_time = t
                                        stable_candidate_count = None
                                        candidate_since = None
                                elif allow_no_person_delivery:
                                    if elapsed >= DELIVERED_STABLE_SECONDS_NO_PERSON:
                                        confirmed_pkg_count = new
                                        print(f"[{time.strftime('%H:%M:%S')}] DELIVERED  (count {old} -> {confirmed_pkg_count}) [no-person]")
                                        send_event("DELIVERED", confirmed_pkg_count, person_present_stable)
                                        last_delivered_time = t
                                        stable_candidate_count = None
                                        candidate_since = None

                        elif new < old:
                            # REMOVED: allow two paths:
                            #  - fast path when a person is present
                            #  - slower path when no person is seen
                            person_recent = (t - person_last_seen) <= REMOVED_PERSON_GRACE_SECONDS
                            allow_no_person = ALLOW_REMOVED_WITHOUT_PERSON

                            if not in_removed_cooldown:
                                if person_present_stable or person_recent:
                                    if elapsed >= REMOVED_STABLE_SECONDS_WITH_PERSON:
                                        confirmed_pkg_count = new
                                        print(f"[{time.strftime('%H:%M:%S')}] REMOVED    (count {old} -> {confirmed_pkg_count})")
                                        send_event("REMOVED", confirmed_pkg_count, person_present_stable)
                                        last_removed_time = t
                                        stable_candidate_count = None
                                        candidate_since = None
                                elif allow_no_person:
                                    if elapsed >= REMOVED_STABLE_SECONDS_NO_PERSON:
                                        confirmed_pkg_count = new
                                        print(f"[{time.strftime('%H:%M:%S')}] REMOVED    (count {old} -> {confirmed_pkg_count}) [no-person]")
                                        send_event("REMOVED", confirmed_pkg_count, person_present_stable)
                                        last_removed_time = t
                                        stable_candidate_count = None
                                        candidate_since = None
            else:
                stable_candidate_count = None
                candidate_since = None

        if SHOW_WINDOW:
            cv2.imshow("package_events_person", frame_vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
