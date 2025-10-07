import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from datetime import datetime
import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

# Initialize the DeepSort tracker
object_tracker = DeepSort(max_age=10,
                          n_init=2,
                          nms_max_overlap=0.4,
                          max_cosine_distance=0.3,
                          nn_budget=None,
                          override_track_class=None,
                          embedder="mobilenet",
                          half=True,
                          bgr=True,
                          embedder_gpu=True,
                          embedder_model_name=None,
                          embedder_wts=None,
                          polygon=False,
                          today=None)

img_size = 640

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # or yolov5n - yolov5x6, custom
model.cuda()

# Confidence threshold
confidence_threshold = 0.25
model.conf = confidence_threshold

# Webcam index {0: Laptop webcam, 1: Ivcam, 2: Iriun, 3: Eopccam}
# cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture('test_object_tracking.mp4')
# Set the resolution of the webcam to specific size


def prediction(img):
    ################################################
    # Find the objects and their coordinates by YOLO
    ################################################
    check = False
    results = model(img)

    if not results.pandas().xyxyn[0].empty:
        check = True

    return check, results


def yolo_to_coco(x_center, y_center, w, h, image_width, image_height):
    #########################################################
    # Convert normalized yolo coordinates to coco coordinates
    #########################################################

    width = int(w * image_width)
    height = int(h * image_height)
    x = int(((2 * x_center * image_width) - w) / 2)
    y = int(((2 * y_center * image_height) - h) / 2)

    cord = [x, y, width, height]

    return cord


def plot_boxes(results, frame, height, width):
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    detections = []

    n = len(labels)
    x_shape, y_shape = width, height

    for i in range(n):
        row = cord[i]

        x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
            row[3] * y_shape)

        x_center = x1 + (x2 - x1)
        y_center = y1 + ((y2 - y1) / 2)

        tlwh = np.asarray([x1, y1, int(x2 - x1), int(y2 - y1)], dtype=np.float32)
        confidence = float(row[4].item())

        detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], row[4].item(), labels[i]))

    return frame, detections


if cap.isOpened():
    window_handle = cv2.namedWindow("Car damage detection", cv2.WINDOW_AUTOSIZE)
    # Window
    while cv2.getWindowProperty("Car damage detection", 0) >= 0:
        ret, frame = cap.read()
        start = time.perf_counter()
        if ret:
            # detection process
            check, results = prediction(frame)
            cord = []
            if check:
                frame, detections = plot_boxes(results, frame, height=frame.shape[0], width=frame.shape[1])

                tracks = object_tracker.update_tracks(detections,
                                                      frame=frame)  # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
                for j, track in enumerate(tracks):
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    ltrb = track.to_ltrb()

                    bbox = ltrb

                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                    cv2.putText(frame, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            end = time.perf_counter()
            totalTime = end - start
            fps = 1 / totalTime
            cv2.putText(frame, "FPS: " + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
            # Show results
            cv2.imshow("Car damage detection", frame)
            keyCode = cv2.waitKey(30)

            if keyCode == ord('q'):
                break
else:
    print("Unable to open camera")

cap.release()
# out.release()
cv2.destroyAllWindows()
