import cv2
from ultralytics import YOLO
import random

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can switch to a custom or more powerful model

# Start webcam
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
treasure_obj = None
treasure_box = None
treasure_index = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    names = results.names

    # Filter out "person" class objects
    filtered_indices = [i for i, c in enumerate(classes) if names[int(c)] != "person"]

    if treasure_obj is None and filtered_indices:
        treasure_index = random.choice(filtered_indices)
        treasure_obj = names[int(classes[treasure_index])]
        treasure_box = boxes[treasure_index]

    for i, box in enumerate(boxes):
        label = names[int(classes[i])]
        x1, y1, x2, y2 = map(int, box)

        # Color logic
        if i == treasure_index:
            color = (0, 255, 255)  # Yellow for treasure
            cv2.putText(frame, f"Treasure: {label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            color = (0, 255, 0) if label != "person" else (255, 0, 0)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Clue system
    if treasure_box is not None:
        h, w = frame.shape[:2]
        center_player = (w // 2, h // 2)
        tx = int((treasure_box[0] + treasure_box[2]) / 2)
        ty = int((treasure_box[1] + treasure_box[3]) / 2)
        center_treasure = (tx, ty)

        distance = ((center_player[0] - tx)**2 + (center_player[1] - ty)**2)**0.5

        if distance < 50:
            clue = "You found it!"
        elif distance < 150:
            clue = "You're very close!"
        elif distance < 300:
            clue = "Getting warmer..."
        else:
            clue = "Too far!"

        cv2.putText(frame, clue, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

    cv2.imshow("Treasure Hunt", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
