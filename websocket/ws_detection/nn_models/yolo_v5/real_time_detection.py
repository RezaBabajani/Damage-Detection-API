import os
from datetime import datetime
import cv2
import numpy as np
import torch

save_path = "./real_time_detection/"
img_size = 640

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # or yolov5n - yolov5x6, custom
model.cuda()

# Confidence threshold
confidence_threshold = 0.25
model.conf = confidence_threshold

# Webcam index {0: Laptop webcam, 1: Ivcam, 2: Iriun, 3: Eopccam}
cap = cv2.VideoCapture(0)
# Set the resolution of the webcam to specific size
cap.set(3, img_size)  # Width
cap.set(4, img_size)  # height

now = datetime.now()
# dd/mm/YY H:M
dt_string = now.strftime("%d_%m_%Y %H_%M")
# Set resolution for saving the video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
out = cv2.VideoWriter(save_path + dt_string + '.avi',
                      cv2.VideoWriter_fourcc(*'MJPG'),
                      3, size)


def prediction(img):
    ################################################
    # Find the objects and their coordinates by YOLO
    ################################################
    check = False
    prediction = model(img)
    results = prediction.pandas().xyxyn[0]  # Results of the prediction (coordinates, confidence, labels)

    if not results.empty:
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


if cap.isOpened():
    window_handle = cv2.namedWindow("Car damage detection", cv2.WINDOW_AUTOSIZE)
    # Window
    while cv2.getWindowProperty("Car damage detection", 0) >= 0:
        ret, frame = cap.read()
        if ret:
            print(frame)
            # detection process
            check, results = prediction(frame)

            if check:
                for i in range(0, len(results)):
                    # Extract coordinates
                    x_center = results['xmin'][i]
                    y_center = results['ymin'][i]
                    w = results['xmax'][i]
                    h = results['ymax'][i]  # YOLO (normalized) coordinates of box

                    # Convert the coordinate to coco
                    cord = yolo_to_coco(x_center, y_center, w, h, frame_width, frame_height)
                    frame = cv2.rectangle(frame, (cord[0], cord[1]), (cord[2], cord[3]), (36, 255, 12), 1)
                    cv2.putText(frame, results["name"][i], (cord[0], cord[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

            # Show results
            cv2.putText(frame, f"FPS: {cap.get(cv2.CAP_PROP_FPS)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        2)
            cv2.imshow("Car damage detection", frame)
            keyCode = cv2.waitKey(30)
            if keyCode == ord('q'):
                break
else:
    print("Unable to open camera")

cap.release()
# out.release()
cv2.destroyAllWindows()
