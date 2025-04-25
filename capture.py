import cv2
import math
import random
import numpy as np
from ultralytics import YOLO

# Helper functions
def centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def match_box(original_box, detections, names, target_class, max_dist=100):
    cx1, cy1, cx2, cy2 = original_box
    center_orig = centroid((cx1, cy1, cx2, cy2))
    best_box = None
    best_dist = float('inf')

    for x1, y1, x2, y2, conf, cls in detections:
        label = names[int(cls)]
        if label != target_class:
            continue
        center_new = centroid((x1, y1, x2, y2))
        dist = euclidean(center_orig, center_new)
        if dist < best_dist and dist < max_dist:
            best_dist = dist
            best_box = (x1, y1, x2, y2)
    return best_box

def draw_box(frame, box, label, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    detector = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)

    print("🎮 Press 'c' to capture, 'v' to validate, 'q' to quit")

    hunter_box = None
    treasure_box = None
    treasure_cls = None
    model_names = detector.names
    THRESHOLD = 100  # px

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c'):
            print("📸 Capturing...")
            results = detector(frame)[0]
            detections = results.boxes.data.tolist()

            person_boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, conf, cls in detections if model_names[int(cls)] == "person"]
            object_boxes = [(x1, y1, x2, y2, int(cls)) for x1, y1, x2, y2, conf, cls in detections if model_names[int(cls)] != "person"]

            if not person_boxes or not object_boxes:
                print("❌ Need at least one person and one non-person object.")
                continue

            # Randomly assign hunter and treasure
            hunter_box = random.choice(person_boxes)
            treasure_box, treasure_cls = random.choice(object_boxes)[:4], object_boxes[0][4]

            print("🎯 Treasure and Hunter randomly assigned.")

        elif key == ord('v') and hunter_box and treasure_box:
            print("🔎 Validating positions...")
            results = detector(frame)[0]
            detections = results.boxes.data.tolist()

            matched_hunter = match_box(hunter_box, detections, model_names, target_class="person")
            matched_treasure = match_box(treasure_box, detections, model_names, target_class=model_names[treasure_cls])
            print(f"Matched Hunter: {matched_hunter}, Matched Treasure: {matched_treasure}")
            if matched_hunter and matched_treasure:
                c1 = centroid(matched_hunter)
                c2 = centroid(matched_treasure)
                dist = euclidean(c1, c2)
                print(f"📏 Distance: {dist:.2f} px")
                if dist < THRESHOLD:
                    print("🏆 Treasure found!")
                else:
                    print("❌ Not close enough.")
            else:
                print("⚠️ Could not match hunter or treasure.")

        # Visualize latest boxes if available
        if hunter_box:
            draw_box(frame, hunter_box, "Hunter (original)", (255, 0, 0))
        if treasure_box:
            draw_box(frame, treasure_box, "Treasure (original)", (0, 255, 0))

        cv2.imshow("Treasure Hunt Capture Game", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
