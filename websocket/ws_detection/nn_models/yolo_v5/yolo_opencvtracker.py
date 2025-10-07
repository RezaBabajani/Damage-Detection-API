import os
from datetime import datetime
import cv2
import numpy as np
import torch

img_size = 640

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # or yolov5n - yolov5x6, custom
model.cuda()

# Initialize variables
prev_boxes = None
obj_id = 0
object1 = None

# Set confidence threshold
confidence_threshold = 0.25
model.conf = confidence_threshold

# Set NMS threshold
nms_threshold = 0.4
model.nms_thresh = nms_threshold

# Webcam index {0: Laptop webcam, 1: Ivcam, 2: Iriun, 3: Eopccam}
cap = cv2.VideoCapture(0)
# Set the resolution of the webcam to specific size
cap.set(3, img_size)  # Width
cap.set(4, img_size)  # height
MAX_FEATURES = 100


def calculate_transformation(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * 1)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Apply homography to first image
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    # Overlap the two images
    imReg = cv2.addWeighted(im1Reg, 0.5, im2, 0.5, 0)

    return h, mask


def apply_transformation(box, h_transformation):
    """
            Apply the homography to the box coordinates

            Args:
                box (list): [x, y, w, h]
            Returns:
                list: [x, y, w, h]
            """
    x0 = box[0]
    y0 = box[1]
    x1 = box[0] + box[2]
    y1 = box[1] + box[3]
    new_box = []
    for p in [[x0, y0], [x1, y0], [x0, y1], [x1, y1]]:
        p = np.array([p[0], p[1], 1])
        new_p = h_transformation @ p
        new_p = new_p / new_p[2]
        new_box.append(new_p[:2])
    w = (new_box[1][0] - new_box[0][0] + new_box[3][0] - new_box[2][0]) / 2
    h = (new_box[2][1] - new_box[0][1] + new_box[3][1] - new_box[1][1]) / 2
    x0 = (new_box[0][0] + new_box[2][0]) / 2
    y0 = (new_box[0][1] + new_box[1][1]) / 2
    return [int(x0), int(y0), int(w), int(h)]


# Define function to draw boxes around detected objects
def draw_boxes(img, boxes, cls):
    # Convert the boxes tensor into numpy array
    x1, y1, w, h = boxes
    cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
    cv2.putText(frame, cls, (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# Initialize dictionary for storing object IDs and bounding boxes
tracked_objects = {}
old_defaut = []
ref_img = None
index = 0
# Main loop
while True:
    # Capture frame from camera
    ret, frame = cap.read()

    if index == 0:
        ref_img = frame.copy()

    # Calculate homography
    h, mask = calculate_transformation(ref_img, frame.copy())
    # Apply YOLO detection to the current frame
    results = model(frame)
    old_defaut_corrected = []

    for cord, cls in old_defaut:
        cord = apply_transformation(cord, h)
        draw_boxes(frame, cord, cls)
        old_defaut_corrected.append([cord, cls])
    old_defaut = old_defaut_corrected

    if index % 30 == 0:
        old_defaut = []
        print("reset")
        for box in results.xywh[0].cpu().numpy():
            cord = box[:-2]
            cls = results.names[int(box[-1])]
            old_defaut.append([cord, cls])
    # Show frame
    cv2.imshow('Frame', frame)

    index += 1
    # Set current frame as reference image for next iteration
    ref_img = frame.copy()
    # Check for key press
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
