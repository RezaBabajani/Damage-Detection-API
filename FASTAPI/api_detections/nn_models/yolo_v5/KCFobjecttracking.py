import os
from datetime import datetime
import cv2
import numpy as np
import torch

img_size = 640

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # or yolov5n - yolov5x6, custom
model.cuda()

# Confidence threshold
confidence_threshold = 0.25
model.conf = confidence_threshold

# Webcam index {0: Laptop webcam, 1: Ivcam, 2: Iriun, 3: Eopccam}
# cap = cv2.VideoCapture(0)
# Read a video from file
cap = cv2.VideoCapture('test_object_tracking.mp4')
# Set the resolution of the webcam to specific size
cap.set(3, img_size)  # Width
cap.set(4, img_size)  # height


def compute_iou(bbox1, bbox2):
    """Compute intersection over union (IoU) between two bounding boxes.

    Args:
        bbox1 (list): Bounding box coordinates in the format [xmin, ymin, xmax, ymax].
        bbox2 (list): Bounding box coordinates in the format [xmin, ymin, xmax, ymax].

    Returns:
        float: IoU between the two bounding boxes.
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    iou = intersection / union
    return iou


# initialize the object tracker
tracker = cv2.MultiTracker_create()

# Initialize CSRT tracker
# tracker = cv2.TrackerCSRT_create()
save_damage = []
frame_number = 0

# Process first frame
ret, frame = cap.read()
detections = model(frame, size=img_size)

for *xyxy, conf, cls in detections.xyxy[0].cpu().numpy():
    damage_type = detections.names[int(cls)]
    bbox = (xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])

    tracker.add(cv2.TrackerCSRT_create(), frame, bbox)
    save_damage.append([*xyxy, damage_type])

if cap.isOpened():
    window_handle = cv2.namedWindow("Car damage detection", cv2.WINDOW_AUTOSIZE)
    # Window
    while cv2.getWindowProperty("Car damage detection", 0) >= 0:
        ret, frame = cap.read()
        if ret:
            # try:
            # Update tracker
            success, boxes = tracker.update(frame)
            for ind, box in enumerate(boxes):
                (x, y, w, h) = [int(v) for v in box]
                width = save_damage[ind][2]
                height = save_damage[ind][3]
                # Remove the box from tracker if it is out of box width of height
                if save_damage[ind][0] + width < x or save_damage[ind][1] + height < y or x + w < save_damage[ind][0] or y + h < save_damage[ind][1]:
                    # Remove the tracker of the box
                    # tracker.remove(ind)
                    # Remove the box from save_damage
                    # save_damage.pop(ind)
                    continue
                else:
                    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
                    save_damage[ind][:4] = box

            # If the frame_number is divisible by 10, then run YOLOv5
            if frame_number % 20 == 0:
                results = model(frame, size=img_size)
                for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
                    damage_type = results.names[int(cls)]
                    bbox = (xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])
                    # Check if there is any newly detected box with IOU more than 0.5 with saved boxes
                    for i, saved_box in enumerate(save_damage):
                        iou = compute_iou(bbox, saved_box[:4])
                        if iou < 0.5:
                            tracker.add(cv2.TrackerCSRT_create(), frame, bbox)
                            save_damage.append([*xyxy, damage_type])
                            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                                          (int(bbox[2] + bbox[0]), int(bbox[3] + bbox[1])), (0, 255, 0), 2)

                            # cv2.putText(frame, saved_box[4], (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX,
                            #             1,
                            #             (0, 255, 0), 2)
                            break

            # except:
            #     pass
            # Put framerate on the frame
            cv2.putText(frame, f"Frame number: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Car damage detection', frame)
            if cv2.waitKey(1) == 27:
                break
            frame_number += 1
else:
    print("Unable to open camera")

cap.release()
# out.release()
cv2.destroyAllWindows()
