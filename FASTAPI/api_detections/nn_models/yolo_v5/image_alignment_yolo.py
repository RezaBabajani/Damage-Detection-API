import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

img_size = 640

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # or yolov5n - yolov5x6, custom
model.cuda()

# Set confidence threshold
confidence_threshold = 0.25
model.conf = confidence_threshold

# Initialize the camera
cap = cv2.VideoCapture(0)

method = "ORB"

# Initialize the feature detector and matcher using FLANN
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1)
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

if method == "ORB":
    # Initialize the feature detector and matcher by ORB
    detector = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_HARRIS_SCORE)
    matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
elif method == "SIFT":
    # Initialize the feature detector and matcher
    detector = cv2.SIFT_create()
    # matcher = cv2.BFMatcher_create(cv2.NORM_L2)

# Initialize the previous frame and previous keypoints
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_kp, prev_des = detector.detectAndCompute(prev_gray, None)

# Initialize the transformation matrix
M = np.eye(3)

# Initialize the detected boxes
damages = []


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

# create a figure and axes
fig, ax = plt.subplots()

# create a blank image
img = np.zeros((480, 640, 3), dtype=np.uint8)

# create a plot with the blank image
im = ax.imshow(img)

# show the plot
plt.show(block=False)

while True:
    # Capture a new frame
    ret, frame = cap.read()
    width, height, _ = frame.shape
    # print(width, height)
    if not ret:
        break

    # Detect the damages by the YOLO model
    results = model(frame, size=img_size)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors for the current frame
    curr_kp, curr_des = detector.detectAndCompute(gray, None)

    if damages:
        # Match keypoints between the previous and current frames
        # matches = matcher.match(prev_des, curr_des, crossCheck=True)

        matches = matcher.match(prev_des, curr_des)

        # Select only good matches
        good_matches = [match for match in matches if match.distance < 300]

        # Calculate the transformation matrix using the good matches
        src_pts = np.float32([prev_kp[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([curr_kp[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
        # Calculate the displacement of the camera in x and y directions in pixels
        if M is not None:
            dx = (M[0, 2] * (img_size / width))
            dy = (M[1, 2] * (img_size / width))
        else:
            dx = 0
            dy = 0

        # if dx and dy are more than 5 pixels, then update the previous frame and previous keypoints
        if abs(dx) > 5 or abs(dy) > 5:
            # Update the damage boxes according to the displacement
            for damage in damages:
                # If the x of damage in more or less than of hald damage width, then remove the damage
                if dx > (damage[2]) or dy < (damage[3]):
                    damages.remove(damage)
                    continue
                else:
                    damage[0] -= dx
                    damage[1] -= dy
                    damage[2] -= dx
                    damage[3] -= dy

        # Check if the detected boxes coordinates has the overlap more than 0.5 with the damage boxes
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            for damage in damages:
                if compute_iou(damage, [*xyxy]) > 0.1:
                    overlap = True
                    break
                else:
                    overlap = False
            if not overlap:
                damage_type = results.names[int(cls)]
                damages.append([*xyxy, damage_type])

    else:
        # If there is no damage, set the displacement to zero
        dx = 0
        dy = 0
        # Save the detected boxes coordinates and labels in the damages list
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            # Add cls and xyxy to the damages list
            damage_type = results.names[int(cls)]
            damages.append([*xyxy, damage_type])

    # Draw rectangles around the damages list and put the cls on top left of that
    for damage in damages:
        cv2.rectangle(frame, (int(damage[0]), int(damage[1])), (int(damage[2]), int(damage[3])), (0, 0, 255), 2)
        cv2.putText(frame, damage[4], (int(damage[0]), int(damage[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    2, cv2.LINE_AA)

    # Draw the tracked keypoints on the current frame
    # matched_img = cv2.drawMatches(prev_gray, prev_kp, gray, curr_kp, good_matches, None)

    # Update the previous keypoints and frame
    prev_gray = gray.copy()
    prev_kp = curr_kp
    prev_des = curr_des

    # Display the current frame and the displacement values
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'dx={:.2f}, dy={:.2f}'.format(dx, dy), (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # # update the image
    # im.set_data(frame)
    #
    # # redraw the plot
    # fig.canvas.draw()
    #
    # # wait for a short time
    # plt.pause(0.001)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()
