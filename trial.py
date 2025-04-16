# treasure_hunt_with_reselect.py
# ---------------------------------------
# Prerequisites:
#   pip install opencv-python ultralytics deep_sort_realtime numpy
#
# Usage:
#   python treasure_hunt_with_reselect.py
# ---------------------------------------

import cv2
import math
import random
import numpy as np
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
    # sort: treasure first, then others
    order = []
    if treasure_id in known_ids:
        order.append(treasure_id)
    order += [oid for oid in sorted(known_ids) if oid != treasure_id]
    y = 60
    for oid in order:
        color = (0,255,0) if oid == treasure_id else (200,200,200)
        cv2.putText(sidebar, f"ID {oid}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y +=  30
    # append sidebar to the right of the frame
    return np.hstack([frame, sidebar])

def main():
    # ——— Config ———
    DIST_THRESHOLD = 50    # in pixels
    CAMERA_INDEX = 0
    # -------------- 

    detector = YOLO('yolov8n.pt')
    tracker  = DeepSort(max_age=30, n_init=3)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    print("📷  cap.isOpened():", cap.isOpened())
    if not cap.isOpened():
        print("❌ Cannot open camera"); 
        

    treasure_id = None
    hunter_id   = None
    known_obj_ids = set()
    state = "WAIT_OBJECT"

    print("🔍 Waiting for non-person object to pick treasure...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1) Detection
        results = detector(frame)[0]
        dets_for_tracker = []
        for b in results.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = float(b.conf[0])
            cls  = int(b.cls[0])
            # DeepSort wants tlwh format:
            dets_for_tracker.append(([x1, y1, w, h], conf, cls))

        # 3) Tracking
        tracks = tracker.update_tracks(dets_for_tracker, frame=frame)

        # build current lists
        curr_objs = [t for t in tracks if t.is_confirmed() and t.det_class != 0]
        curr_persons = [t for t in tracks if t.is_confirmed() and t.det_class == 0]

        # update known objects
        for t in curr_objs:
            known_obj_ids.add(t.track_id)

        # 3) State logic & keypress
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):       # quit on 'q'
            print("🛑 Quitting.")
            break
        elif key == ord('r'):
            if curr_objs:
                treasure_id = random.choice([t.track_id for t in curr_objs])
                state = "HUNTING" if hunter_id else "WAIT_HUNTER"
                print(f"🔄 Reselected treasure → ID {treasure_id}")
        elif key == 27:  # ESC
            break

        # auto‐pick treasure if not set and objects exist
        if treasure_id is None and curr_objs:
            treasure_id = random.choice([t.track_id for t in curr_objs])
            state = "WAIT_HUNTER"
            print(f"🎯 Auto-picked treasure → ID {treasure_id}")

        # assign hunter if not set
        if hunter_id is None and curr_persons:
            hunter_id = curr_persons[0].track_id
            print(f"🏃 Hunter acquired → ID {hunter_id}")
            if treasure_id: state = "HUNTING"

        # if both set, compute distance
        dist = None
        if state == "HUNTING":
            box_t = next((t.to_ltrb() for t in tracks if t.track_id==treasure_id and t.is_confirmed()), None)
            box_h = next((t.to_ltrb() for t in tracks if t.track_id==hunter_id   and t.is_confirmed()), None)
            if box_t.any() and box_h.any():
                c_t, c_h = centroid(box_t), centroid(box_h)
                dist = euclidean(c_t, c_h)
                if dist < DIST_THRESHOLD:
                    print("🏆 Treasure Found!")
                    state = "FOUND"

        # 4) Visualization
        for t in tracks:
            if not t.is_confirmed(): continue
            x1,y1,x2,y2 = map(int, t.to_ltrb())
            tid = t.track_id
            if tid == treasure_id:
                col, lbl = (0,255,0), f"T:{tid}"
            elif tid == hunter_id:
                col, lbl = (255,0,0), f"H:{tid}"
            else:
                col, lbl = (200,200,200), f"{t.det_class}:{tid}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
            cv2.putText(frame, lbl, (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        # overlay info
        cv2.putText(frame, f"State: {state}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        if dist is not None:
            cv2.putText(frame, f"Dist: {dist:.1f}px", (10,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # draw sidebar of known objects
        out = draw_sidebar(frame, known_obj_ids, treasure_id)

        cv2.imshow("Treasure Hunt", out)

        if state == "FOUND":
            cv2.waitKey(0)
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()