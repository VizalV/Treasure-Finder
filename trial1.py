# treasure_hunt_show_bboxes.py
# ---------------------------------------
# Prerequisites:
#   pip install opencv-python ultralytics deep_sort_realtime numpy torch
#
# Usage:
#   python treasure_hunt_show_bboxes.py
# ---------------------------------------

import cv2
import math
import random
import numpy as np
import datetime
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def draw_sidebar(frame, known_ids, treasure_id):
    h, w = frame.shape[:2]
    panel_w = 200
    sidebar = np.zeros((h, panel_w, 3), dtype=np.uint8)
    cv2.putText(sidebar, "Objects:", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
    order = []
    if treasure_id in known_ids:
        order.append(treasure_id)
    order += [oid for oid in sorted(known_ids) if oid != treasure_id]
    y, line_h = 60, 30
    for oid in order:
        color = (0,255,0) if oid == treasure_id else (200,200,200)
        cv2.putText(sidebar, f"ID {oid}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += line_h
    return np.hstack([frame, sidebar])

def main():
    # ——— Config ———
    DIST_THRESHOLD = 50    # pixels
    CAMERA_INDEX   = 0
    DETECT_EVERY   = 3     # run detection every N frames
    RESIZE_DIM     = 320   # detector input size
    # --------------

    # Device & model setup
    if torch.cuda.is_available():
        device = 'cuda:0'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    detector = YOLO('yolov8n.pt')
    detector.model.fuse()
    if device.startswith(device):
        detector.model.half()

    tracker = DeepSort(max_age=30, n_init=3, embedder_gpu=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    print("📷 cap.isOpened():", cap.isOpened())
    if not cap.isOpened():
        return

    treasure_id   = None
    hunter_id     = None
    known_obj_ids = set()
    state         = "WAIT_OBJECT"
    prev_time     = datetime.datetime.now()
    frame_idx     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h0, w0 = frame.shape[:2]

        # 1) Detection / Tracking
        if frame_idx % DETECT_EVERY == 0:
            small   = cv2.resize(frame, (RESIZE_DIM, RESIZE_DIM))
            results = detector(small)[0]
            scale_x = w0 / RESIZE_DIM
            scale_y = h0 / RESIZE_DIM
            dets    = []
            for x1, y1, x2, y2, conf, cls in results.boxes.data.tolist():
                x1o = int(x1 * scale_x)
                y1o = int(y1 * scale_y)
                wo  = int((x2 - x1) * scale_x)
                ho  = int((y2 - y1) * scale_y)
                dets.append(([x1o, y1o, wo, ho], conf, int(cls)))
            tracks = tracker.update_tracks(dets, frame=frame)
        else:
            tracks = tracker.update_tracks([], frame=frame)

        # list current
        curr_objs    = [t for t in tracks if t.is_confirmed() and t.det_class != 0]
        curr_persons = [t for t in tracks if t.is_confirmed() and t.det_class == 0]
        for t in curr_objs:
            known_obj_ids.add(t.track_id)

        # 2) Keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r') and curr_objs:
            treasure_id = random.choice([t.track_id for t in curr_objs])
            state = "HUNTING" if hunter_id else "WAIT_HUNTER"
        if key == ord('h') and curr_persons:
            hunter_id = random.choice([t.track_id for t in curr_persons])
            state = "HUNTING" if treasure_id else state

        # auto‐assign
        if treasure_id is None and curr_objs:
            treasure_id = random.choice([t.track_id for t in curr_objs])
            state = "WAIT_HUNTER"
        if hunter_id is None and curr_persons:
            hunter_id = curr_persons[0].track_id
            state = "HUNTING" if treasure_id else state

        # 3) Compute distance & grab bboxes
        box_t = next((t.to_ltrb() for t in tracks
                      if t.track_id == treasure_id and t.is_confirmed()), None)
        box_h = next((t.to_ltrb() for t in tracks
                      if t.track_id == hunter_id   and t.is_confirmed()), None)
        dist = None
        if state == "HUNTING" and box_t and box_h:
            c_t  = centroid(box_t)
            c_h  = centroid(box_h)
            dist = euclidean(c_t, c_h)
            if dist < DIST_THRESHOLD:
                state = "FOUND"

        # print coords every frame
        if box_t:
            print(f"Treasure {treasure_id} bbox:", tuple(map(int, box_t)))
        if box_h:
            print(f"Hunter   {hunter_id} bbox:", tuple(map(int, box_h)))

        # 4) Draw only hunter & treasure
        if box_t:
            x1,y1,x2,y2 = map(int, box_t)
            cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"T:{treasure_id}", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        if box_h:
            x1,y1,x2,y2 = map(int, box_h)
            cv2.rectangle(frame, (x1,y1),(x2,y2), (255,0,0), 2)
            cv2.putText(frame, f"H:{hunter_id}", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # overlays
        cv2.putText(frame, f"State: {state}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        if dist is not None:
            cv2.putText(frame, f"Dist: {dist:.1f}px", (10,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        out = draw_sidebar(frame, known_obj_ids, treasure_id)

        now = datetime.datetime.now()
        fps = 1.0 / (now - prev_time).total_seconds()
        prev_time = now
        cv2.putText(out, f"FPS: {fps:.1f}", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

        cv2.imshow("Treasure Hunt", out)
        frame_idx += 1

        if state == "FOUND":
            cv2.waitKey(0)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()