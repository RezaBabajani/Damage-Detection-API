import os
from datetime import datetime
import cv2
import numpy as np
import torch
from sort import *

img_size = 640

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # or yolov5n - yolov5x6, custom
model.cuda()

sort_tracker = Sort(max_age=1,
                    min_hits=2,
                    iou_threshold=0.4)

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

"""Function to Draw Bounding boxes"""


def draw_boxes(img, bbox, identities=None, categories=None, confidences=None, names=None, colors=None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        tl = 2 or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        # conf = confidences[i] if confidences is not None else 0

        color = colors[cat]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

        label = str(id) + ":" + names[cat] if identities is not None else f'{names[cat]} {confidences[i]:.2f}'
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return img

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

    save_damage.append([*xyxy, damage_type])

if cap.isOpened():
    window_handle = cv2.namedWindow("Car damage detection", cv2.WINDOW_AUTOSIZE)
    # Window
    while cv2.getWindowProperty("Car damage detection", 0) >= 0:
        ret, frame = cap.read()
        if ret:
            # If the frame_number is divisible by 10, then run YOLOv5
            results = model(frame, size=img_size)
            dets_to_sort = np.empty((0, 6))
            for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
                dets_to_sort = np.vstack((dets_to_sort,
                                          np.array([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]),
                                                    float(conf), results.names[int(cls)]])))

                # Draw the detection box on frames
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                # cv2.putText(frame, results.names[int(cls)], (int(xyxy[0]), int(xyxy[1]) - 5),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)




                # damage_type = results.names[int(cls)]
                # bbox = (xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])
                # Check if there is any newly detected box with IOU more than 0.5 with saved boxes
            tracked_dets = sort_tracker.update(dets_to_sort)
            tracks = sort_tracker.getTrackers()

            # draw boxes for visualization
            if len(tracked_dets) > 0:
                bbox_xyxy = tracked_dets[:, :4]
                identities = tracked_dets[:, 8]
                categories = tracked_dets[:, 4]
                confidences = None

                for ind, box in enumerate(bbox_xyxy):
                    # Draw on frames
                    x1 = int(float(box[0]))
                    y1 = int(float(box[1]))
                    x2 = int(float(box[2]))
                    y2 = int(float(box[3]))
                    # Draw bounding box on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw text on frame
                    # cv2.putText(frame, categories[ind], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    #             (0, 255, 0), 2)

            # frame = draw_boxes(frame, bbox_xyxy, identities, categories, confidences, names, colors)

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
