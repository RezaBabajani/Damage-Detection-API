import cv2
import numpy as np
import torch

img_size = 640

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # or yolov5n - yolov5x6, custom
model.cuda()

# Set confidence threshold
confidence_threshold = 0.25
model.conf = confidence_threshold

# Create a video capture object
cap = cv2.VideoCapture(0) # use 0 for the default camera or a file path for a video file

# Create a Lucas-Kanade optical flow object
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize variables for optical flow
prev_frame = None
prev_points = None

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


while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Detect the damages by the YOLO model
    results = model(frame, size=img_size)
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is not None and prev_points is not None:

        if damages:
            # Compute optical flow using Lucas-Kanade algorithm
            next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, gray, prev_points, None, **lk_params)

            # Select good points
            good_points = next_points[status == 1]
            prev_points = prev_points[status == 1]

            # Compute camera motion parameters
            E, mask = cv2.findEssentialMat(prev_points, good_points, np.eye(3), cv2.RANSAC, 0.999, 1.0, None)
            _, R, t, _ = cv2.recoverPose(E, prev_points, good_points)

            # Find dx and dy
            dx = t[0][0]
            dy = t[1][0]

            # Update the damage boxes according to the displacement
            for damage in damages:
                # If the x of damage in more or less than of hald damage width, then remove the damage
                if (damage[0] + dx) > (0.5 * damage[2] + damage[0]) or (damage[0] + damage[2] + dx) < (
                        0.5 * damage[2] + damage[0]):
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

            # Print camera motion parameters
            # print('Rotation:\n', R)
            # print('Translation:\n', t)
            # # Show camera motion parameters on frame
            # cv2.putText(frame, 'Rotation: ' + str(R), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # cv2.putText(frame, 'Translation: ' + str(t), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        # If there is no damage, set the displacement to zero
        dx = 0
        dy = 0
        # Save the detected boxes coordinates and labels in the damages list
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            # Add cls and xyxy to the damages list
            damage_type = results.names[int(cls)]
            damages.append([*xyxy, damage_type])
        # Draw optical flow and camera motion on the frame
        # for i, (prev, next) in enumerate(zip(prev_points, good_points)):
        #     x0, y0 = prev.ravel()
        #     x1, y1 = next.ravel()
        #     frame = cv2.circle(frame, (x0, y0), 5, (0, 255, 0), -1)
        #     frame = cv2.line(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

    # Draw rectangles around the damages list and put the cls on top left of that
    for damage in damages:
        cv2.rectangle(frame, (int(damage[0]), int(damage[1])), (int(damage[2]), int(damage[3])), (0, 0, 255), 2)
        cv2.putText(frame, damage[4], (int(damage[0]), int(damage[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255),
                    2, cv2.LINE_AA)

    # Update previous frame and points
    prev_frame = gray.copy()
    prev_points = cv2.goodFeaturesToTrack(gray, 500, 0.01, 10)

    # Display the current frame and the displacement values
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'dx={:.2f}, dy={:.2f}'.format(dx, dy), (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    # Display the frame
    cv2.imshow('Camera', frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy windows
cap.release()
cv2.destroyAllWindows()