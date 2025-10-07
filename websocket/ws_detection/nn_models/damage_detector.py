# -*- coding: utf-8 -*-
import torch
import os


def get_damage_detector(fake=False):
    if fake:
        return FakeDamageDetector()
    else:
        return YoloDetector()


class FakeDamageDetector():
    def prediction(self, frame):
        return "AA-123-BB"


class YoloDetector():
    def __init__(self):
        self.device = torch.device("cuda")
        script_path = os.path.abspath(__file__)
        script_directory = os.path.dirname(script_path)
        self.model = torch.hub.load(script_directory + r'/yolo_v5', 'custom',
                                    path=script_directory + r'/models/damage_detection.pt', source='local')
        self.model.to(self.device)
        self.model.eval()
        self.gpu_busy = False
        self.imgsize = 640

    def prediction(self, frame, confidence):
        ################################################
        # Find the objects and their coordinates by YOLO
        ################################################
        any_detection = False

        self.gpu_busy = True
        with torch.no_grad():
            prediction = self.model(frame)
        self.gpu_busy = False

        if not prediction.pandas().xyxyn[0].empty:
            any_detection = True
        return any_detection, prediction

    @staticmethod
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

    @staticmethod
    def plot_boxes(results, height, width, confidence):
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        detections = []

        n = len(labels)
        x_shape, y_shape = width, height

        for i in range(n):
            row = cord[i]

            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)

            detection_conf = float(row[4].item())
            if detection_conf >= float(confidence):
                detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], detection_conf, results.names[int(labels[i])]))

        return detections

    def apply_model(self, input_frame, confidence):
        from copy import deepcopy
        """ This funciton will take input_frame and apply 
            YOLO detection """

        confidence_threshold = 0.25
        self.model.conf = confidence_threshold
        any_detection, results = self.prediction(input_frame, confidence)
        detections = []
        if any_detection:

            detections = YoloDetector.plot_boxes(results, height=input_frame.shape[0], width=input_frame.shape[1], confidence=confidence)

        return detections

# if __name__ == '__main__':
#     YoloDetector()

