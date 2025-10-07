import os
import cv2
import numpy as np
import torch
import imutils
from utils.general import xyxy2xywh
from PIL import Image

img_path = input("Enter your image path: ")
save_path = input("Enter your save directory for saving annotations: ")
# save_path = "./real_time_detection/"
img_size = 640
split_quantity = 3


def resize(file, size):
    height, width = file.shape[:2]

    if height > width:
        image = imutils.resize(file, height=size)
        canvas = np.zeros((size, size, 3), np.uint8)

        # compute center offset
        x_center = (size - image.shape[1]) // 2
        y_center = (size - image.shape[0]) // 2

        # copy img image into center of result image
        canvas[y_center:y_center + image.shape[0],
               x_center:x_center + image.shape[1]] = image
    else:
        image = imutils.resize(file, width=size)
        canvas = np.zeros((size, size, 3), np.uint8)

        # compute center offset
        x_center = (size - image.shape[1]) // 2
        y_center = (size - image.shape[0]) // 2

        # copy img image into center of result image
        canvas[y_center:y_center + image.shape[0],
               x_center:x_center + image.shape[1]] = image

    return canvas


# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # or yolov5n - yolov5x6, custom

# for img in os.listdir(img_path):
#     image = cv2.imread(img_path + img)
#
#     h, w = image.shape[:2]
#     width_cutoff = w // split_quantity
#
#     cutoff_img = []
#     for i in range(split_quantity):
#         split_img = image[:, (width_cutoff * i):(width_cutoff * (i + 1))]
#         pic = resize(split_img, img_size)
#         results = model(pic)
#
#         # non_padding_img = np.nonzero(np.squeeze(results.render()))
#         cutoff_img.append(np.squeeze(results.render()))
#
#         # In case of showing the detected result
#         cv2.imshow("Part" + str(i), np.squeeze(results.render()))
#         cv2.waitKey(0)
#
#     complete_img = np.concatenate((cutoff_img[0], cutoff_img[1], cutoff_img[2]), axis=1)
#     cv2.imshow("Complete image", complete_img)
#     cv2.waitKey(0)
gn = torch.tensor(cv2.imread(img_path + "/" + os.listdir(img_path)[0]).shape)[[1, 0, 1, 0]]  # normalization gain whwh
for img in os.listdir(img_path):
    image = cv2.imread(img_path + "/" + img)

    # pic = resize(image, img_size)
    results = model(image)
    det = results.xyxy[0]

    for *xyxy, conf, cls in det:
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh)   # label format
        with open(save_path + "/" + img.split(".")[0] + '.txt', 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

    # In case of showing the detected result
    # cv2.imshow("detection", np.squeeze(results.render()))
    # cv2.waitKey(0)
