import cv2
import numpy as np
from typing import Optional
from unittest.mock import Mock
from .frame_alignement import FrameAlignement
from copy import deepcopy
import jsonpickle

list_damage = ['Dent', 'Scratch', 'Missing part', 'Deformed', 'Backlight', 'Frontlight', 'Crack',
               'Misplaced', 'Varnish', 'Rust', 'Glass']


class BayesianFilter:
    def __init__(self,
                 img_shape=[640, 640],
                 list_labels=list_damage,
                 scale_mask_factor=0.4,
                 discount_factor: float = 0.99,
                 threshold: float = 0.2,
                 box_threshold: float = 0.2,
                 erode_size: float = 2,
                 dilate_size: float = 0,
                 growbox_size: float = 0,
                 connectivity: int = 8,
                 enabled: bool = True,
                 iou_inter_class: float = 0.35,
                 union_method_inter_class: str = 'min',
                 extract_boxes_method: str = 'connected_components',
                 melting_enabled: bool = True,
                 overlap_behavior: str = 'merge',
                 crono: Optional['Chronometer'] = None,
                 use_norm_for_discount: bool = True,
                 **kwargs):
        """
        Args:
            img_shape (tuple): The shape of the image
            list_labels (list): The list of labels to track
            scale_mask (float): The scale of the mask relative to the original image
            discount_factor (float): In [0,1], it multiplies the previous damage mask by this factor at each frame
            threshold (float): The threshold for the damage mask (values below are set to 0)
            box_threshold (float): Before extracting the boxes, the damage mask is thresholded with this value
            erode_size (int): Erode the (scaled) damage mask removing this number of pixels in each direction
            dilate_size (int): Dilate the (scaled) damage mask adding this number of pixels in each direction
            connectivity (int): The connectivity for detecting connected components
            growbox_size (int): Increase the (scaled) bounding box adding this number of pixels in each direction
            enabled (bool): If False, the tracker is disabled
            iou_inter_class (float): The IoU threshold for considering a match between damage boxes of different classes
            union_method_inter_class (str): The method to calculate the union of two damage boxes of different classes.
                                            Can be 'union' or 'min'
            overlap_behavior (str): How to process couples of overlapping boxes with different labels.
                                        Choices are: `merge` for merging the two boxes and `remove` for removing the box with lower confidence
            melting_enabled (bool): If True, the damage boxes are melted at each frame
            extract_boxes_method (str): The method to extract the damage boxes. Can be 'connected_components' or 'kmeans'
            use_norm_for_discount (bool): If True, the norm of the homography matrix is used for calculating the discount factor
            crono (Chronometer): The Chronometer object to use for measuring time. If None, no cronometer is used
            **kwargs: The arguments for the FrameAlignement object
        """
        # Initialise the FrameAlignement object
        self.fa = FrameAlignement(self, img_shape=img_shape, crono=crono, **kwargs)
        self.growbox_size = growbox_size
        self.erode_size = erode_size
        self.dilate_size = dilate_size

        self.list_labels = list_labels
        # Create a dictionary to easily convert labels to integers
        self.dict_labels = {label: i for i, label in enumerate(list_labels)}
        self.n_classes = len(self.list_labels)
        self.img_shape = img_shape
        self.scale_mask_factor = scale_mask_factor
        self.damage_shape = (
            int(self.img_shape[0] * self.scale_mask_factor), int(self.img_shape[1] * self.scale_mask_factor))
        self.reset_damage_mask()
        self.damage_mask_uint8 = None
        self.enabled = enabled
        self.connectivity = connectivity
        self.iou_inter_class = iou_inter_class
        self.union_method_inter_class = union_method_inter_class
        self.extract_boxes_method = extract_boxes_method
        self.melting_enabled = melting_enabled
        self.overlap_behavior = overlap_behavior

        self._discount_factor = discount_factor
        self.use_norm_for_discount = use_norm_for_discount

        self.threshold = threshold
        self.box_threshold = box_threshold

        self.current_frame = -1
        self._h = None
        self._last_h_frame = None
        self._scaling_matrix = None
        # Inherited from FrameAlignement
        self.draw_match_points = self.fa.draw_match_points

        # If no crono is provided, use a mock object (i.e. do nothing)
        if crono is None:
            self.crono = Mock()
        else:
            self.crono = crono

        self.last_melting = None
        # This is just a variable for checking if the frame has been updated pefore updating the prediction
        self.frame_updated = True

    def export_state(self):
        current_state = {
            'damage_mask': jsonpickle.dumps(self.damage_mask),
            'damage_mask_uint8': jsonpickle.dumps(self.damage_mask_uint8),
            'fa': self.fa.export_state(),
            'current_frame': jsonpickle.dumps(self.current_frame),
        }
        return current_state

    def load_state(self, state):
        self.damage_mask = jsonpickle.loads(state['damage_mask'])
        self.fa.load_state(state['fa'])
        self.current_frame = jsonpickle.loads(state['current_frame'])
        self.damage_mask_uint8 = jsonpickle.loads(state['damage_mask_uint8'])

    def reset_state(self):
        self.reset_damage_mask()
        self.fa.reset_state()
        self.current_frame = -1
        self.damage_mask_uint8 = None

    @property
    def discount_factor(self):
        if self.use_norm_for_discount:
            if np.isnan(self.fa.h_norm) or self.fa.h_norm > self.fa.max_h_norm:
                return 0
            return max(0, self._discount_factor - 10 * (self.fa.h_norm ** 2))
        return self._discount_factor

    @property
    def h(self):
        """Return the scaled homography matrix"""
        # Cache the result
        if self._last_h_frame == self.current_frame:
            return self._h
        self._last_h_frame = self.current_frame
        h = self.fa.h
        if h is None:
            self._h = None
            return None
        # Scale the homography matrix
        # Create the scaling matrix (only once)
        if self._scaling_matrix is None:
            self._scaling_matrix = np.array([[self.scale_mask_factor, 0, 0], [0, self.scale_mask_factor, 0], [0, 0, 1]])
            self._scaling_matrix_inv = np.linalg.inv(self._scaling_matrix)

        # Calculate a base change on h
        h_rescaled = self._scaling_matrix @ h @ self._scaling_matrix_inv
        self._h = h_rescaled
        return self._h

    def update_predictions(self, predictions):
        """
        Update the tracker with the new predictions. Remeber: you need to update the frame first!!

        Args:
            predictions (list): The list of predictions for the frame
        """

        # print(f"There are {len(predictions)} unfiltered predictions")
        if not self.frame_updated:
            raise Exception('You need to update the frame first!, call method update_frame()')
        self.last_predictions = predictions
        if not self.enabled:
            return
        self.frame_updated = False

        self.crono.start('Update damage mask')
        # Draw the predictions on a new damage mask
        self.crono.start('New predictions')
        self.new_damage_mask = self.get_new_damage_mask(predictions)
        self.crono.stop('New predictions')

        # Sliced version (more efficient)
        for cls in range(self.n_classes):
            # Apply discount factor (old predictions are less important)
            self.damage_mask[cls] *= self.discount_factor
            self.damage_mask[cls] *= -1
            self.damage_mask[cls] += 1
            self.damage_mask[cls] *= self.new_damage_mask[cls]
            self.damage_mask[cls] *= -1
            self.damage_mask[cls] += 1

            # All values below the threshold are set to 0
            self.damage_mask[cls][self.damage_mask[cls] <= self.threshold] = 0

        # # Vectorised version (less efficient but maybe with cuda it could be faster ?)
        # self.damage_mask *= self.discount_factor
        # self.damage_mask *= -1
        # self.damage_mask += 1
        # self.damage_mask *= self.new_damage_mask
        # self.damage_mask *= -1
        # self.damage_mask += 1
        # # All values below the threshold are set to 0
        # self.damage_mask[self.damage_mask <= self.threshold] = 0

        # Convert all the damage masks to uint8
        self.damage_mask_uint8 = (self.damage_mask * 255).astype(np.uint8)
        self.crono.stop('Update damage mask')

    def update_frame(self, frame):
        """
        Update the tracker with the new frame, calculating the homography matrix

        Args:
            frame (np.array): The new frame
        """
        # Update counters
        self.crono.new_cycle()
        self.current_frame += 1
        self.frame_updated = True
        if not self.enabled:
            return

        self.crono.start('Find homography')
        # Update the frame alignement
        success, msg = self.fa.process(frame)
        self.crono.stop('Find homography')

        if success:
            self.crono.start('Apply homography')
            self.apply_homography(self.h)
            self.crono.stop('Apply homography')
        else:
            # print(msg)
            # Reset the damage mask
            self.reset_damage_mask()

    def find_box_kmeans(self):
        """
        Extract damage boxes making clusters of pixels
        """
        from sklearn.cluster import KMeans

        filtered_predictions = []
        for cls in range(self.n_classes):
            gray = self.clean_mask[cls]
            mask = gray > 0
            rows, cols = gray.shape
            coordinates_grid = np.ones((2, rows, cols), dtype=np.int16)
            coordinates_grid[0] = coordinates_grid[0] * np.array([range(rows)]).T
            coordinates_grid[1] = coordinates_grid[1] * np.array([range(cols)])
            non_zero_coords = np.hstack(
                (coordinates_grid[0][mask].reshape(-1, 1), coordinates_grid[1][mask].reshape(-1, 1)))
            # Extract a subset of 3 elements
            non_zero_coords = non_zero_coords[
                              np.random.choice(non_zero_coords.shape[0], int(len(non_zero_coords) * 0.1),
                                               replace=False), :]

            if len(non_zero_coords) < 10:
                continue
            loss_by_n = []
            # Test different numbers of clusters
            for num_clusters in range(1, 5):
                kmeans = KMeans(n_clusters=num_clusters)
                kmeans.fit(non_zero_coords)
                # Convert kmeans clusters to boxes
                labels = kmeans.predict(non_zero_coords)
                tot_loss = 0
                for cluster_index in range(num_clusters):
                    # Get the coordinates belonging to the current cluster
                    cluster_coords = non_zero_coords[labels == cluster_index]
                    # Find the minimum and maximum x,y coordinates
                    min_y, min_x = np.min(cluster_coords, axis=0)
                    max_y, max_x = np.max(cluster_coords, axis=0)

                    width = int((max_x - min_x) // self.scale_mask_factor)
                    height = int((max_y - min_y) // self.scale_mask_factor)
                    # For each option calculate a loss, depending on the number of clusters and size
                    tot_loss += self.box_loss(width, height)
                loss_by_n.append(tot_loss)
            # print(loss_by_n)
            # Choose the number of clusters which minimises the loss
            best_n = np.argmin(loss_by_n) + 1

            # Perform calculation again with the best number of clusters
            # (I know it's not very efficient but it's not the bottleneck)
            kmeans = KMeans(n_clusters=best_n)
            kmeans.fit(non_zero_coords)
            # Convert kmeans clusters to boxes
            labels = kmeans.predict(non_zero_coords)
            for cluster_index in range(best_n):
                # Get the coordinates belonging to the current cluster
                cluster_coords = non_zero_coords[labels == cluster_index]
                # Find the minimum and maximum x,y coordinates
                min_y, min_x = np.min(cluster_coords, axis=0)
                max_y, max_x = np.max(cluster_coords, axis=0)
                x = int(min_x)
                y = int(min_y)
                w = int((max_x - min_x))
                h = int((max_y - min_y))
                damage = self.damage_mask[cls, y: y + h, x: x + w]
                conf = damage[damage > self.threshold].mean()
                pred = {
                    'x': x // self.scale_mask_factor,
                    'y': y // self.scale_mask_factor,
                    'width': w // self.scale_mask_factor,
                    'height': h // self.scale_mask_factor,
                    'confidence': conf,
                    'labels': self.list_labels[cls],
                }
                filtered_predictions.append(pred)

        return filtered_predictions

    def find_box_connected_components(self):
        filtered_predictions = []
        for cls in range(self.n_classes):
            # TODO: Use connectedComponentsWithStatsWithAlgorithm
            # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(self.clean_mask[cls], 4, cv2.CV_32S)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.clean_mask[cls],
                                                                                    self.connectivity, cv2.CV_32S)
            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]
                # Grow boxes
                x = max(x - self.growbox_size, 0)
                y = max(y - self.growbox_size, 0)
                w = min(w + self.growbox_size, self.damage_shape[1] - x)
                h = min(h + self.growbox_size, self.damage_shape[0] - y)

                damage = self.damage_mask[cls, y: y + h, x: x + w]
                conf = damage[damage > self.threshold].mean()
                pred = {
                    'x': int(x // self.scale_mask_factor),
                    'y': int(y // self.scale_mask_factor),
                    'width': int(w // self.scale_mask_factor),
                    'height': int(h // self.scale_mask_factor),
                    'labels': self.list_labels[cls],
                    'confidence': conf
                }
                filtered_predictions.append(pred)
        return filtered_predictions

    def get_boxes(self):
        """
        Get the boxes from the damage mask using cv2.connectedComponentsWithStats
        """

        if self.current_frame == -1:
            raise Exception('No frame has been processed yet')

        if not self.enabled:
            return self.last_predictions

        self.crono.start('Get boxes')

        # Clean the masks
        kernel_erode = np.ones((self.erode_size, self.erode_size), np.uint8)
        kernel_dilate = np.ones((self.dilate_size, self.dilate_size), np.uint8)
        self.clean_mask = self.damage_mask_uint8.copy()

        # Apply threshold on clean mask
        self.clean_mask[self.clean_mask < (self.box_threshold * 255)] = 0

        for cls in range(self.n_classes):
            self.clean_mask[cls] = cv2.erode(self.clean_mask[cls, :, :], kernel_erode, iterations=1)
            self.clean_mask[cls] = cv2.dilate(self.clean_mask[cls], kernel_dilate, iterations=1)

        # Get the boxes
        if self.extract_boxes_method == 'kmeans':
            filtered_predictions = self.find_box_kmeans()
        elif self.extract_boxes_method == 'connected_components':
            filtered_predictions = self.find_box_connected_components()
        else:
            raise Exception(f'Unknown method {self.extract_boxes_method} for extracting boxes')

        # Remove overlapping boxes of different classes
        filtered_predictions = self.remove_overlapping_boxes(filtered_predictions)

        # Melt boxes in bigger boxes
        if self.melting_enabled:
            # Keep a copy of the predictions before and after melting for debugging
            before_pred = deepcopy(filtered_predictions)
            filtered_predictions = self.melt_boxes(filtered_predictions)
            after_pred = deepcopy(filtered_predictions)
            self.last_melting = {"before": before_pred, "after": after_pred}

        # Remove overlapping boxes of different classes
        filtered_predictions = self.remove_overlapping_boxes(filtered_predictions)

        self.crono.stop('Get boxes')
        # print(f"There are {len(filtered_predictions)} filtered predictions")
        return filtered_predictions

    def box_loss(self, width, height):
        box_cost = width + height
        max_cost = (self.img_shape[0] + self.img_shape[1])
        box_cost_norm = box_cost / max_cost
        box_loss = box_cost_norm ** 1.2 + 0.04
        return box_loss

    def cover_loss(self, boxes):
        """
        Return the loss associated to a covering of boxes
        """
        x_min = min([b['x'] for b in boxes])
        y_min = min([b['y'] for b in boxes])
        x_max = max([b['x'] + b['width'] for b in boxes])
        y_max = max([b['y'] + b['height'] for b in boxes])
        width = x_max - x_min
        height = y_max - y_min
        box_loss = self.box_loss(width, height)
        return box_loss

    @staticmethod
    def melt_covering(boxes):
        """
        Melt all the boxes in the list in a big box
        """
        x_min = min([b['x'] for b in boxes])
        y_min = min([b['y'] for b in boxes])
        x_max = max([b['x'] + b['width'] for b in boxes])
        y_max = max([b['y'] + b['height'] for b in boxes])
        x = x_min
        y = y_min
        width = x_max - x_min
        height = y_max - y_min
        # Calculate the confidence making a weighted mean of the confidences
        # The weight is the geometric mean of the two dimensions. This is because
        # we do not want to give too much importance to big boxes
        confs = [b['confidence'] for b in boxes]
        weights = [(b['width'] * b['height']) ** 0.5 for b in boxes]
        conf = np.average(confs, weights=weights)

        # Take sum of weights for each label
        labels_weights = {}
        for w, b in zip(weights, boxes):
            labels_weights[b['labels']] = labels_weights.get(b['labels'], 0) + w

        label = max(labels_weights, key=labels_weights.get)
        # label = boxes[0]['labels']

        return {
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'labels': label,
            'confidence': conf
        }

    def melt_boxes(self, predictions):
        """
        Melt boxes

        We define some notation:

            - Box: a dictionary with the keys 'x', 'y', 'width', 'height', 'labels', 'confidence', a single detection

            - Covering: a list of same class boxes that could be melted in a single box

            - Partition: a list of coverings, where each box is in exactly one covering

        We calculate a loss function for each covering, and we choose the covering with the minimum loss.
        """
        import itertools as it

        def partition(collection):
            """https://stackoverflow.com/questions/19368375/set-partitions-in-python
            Generator of all the partitions of a list
            """
            if len(collection) == 1:
                yield [collection]
                return

            first = collection[0]
            for smaller in partition(collection[1:]):
                # insert `first` in each of the subpartition's subsets
                for n, subset in enumerate(smaller):
                    yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
                # put `first` in its own subset
                yield [[first]] + smaller

        def partition_loss(partition):
            """
            Calculate the loss of a partition
            """
            cover_costs = [self.cover_loss(cover) for cover in partition]
            loss = sum(cover_costs)
            # print(len(partition), loss, partition, [cover_semiperimeter(cover) for cover in partition])
            return loss
            # return area_norm  * 1 / (1 -  self.coef_box_den * area_norm) * (len(partition) ** self.nbox_loss_power)

        melted_boxes = []
        for cls in range(self.n_classes):
            # Get the boxes of the current class
            boxes = [p for p in predictions if p['labels'] == self.list_labels[cls]]
            if len(boxes) == 1:
                melted_boxes.append(boxes[0])
                continue
            elif len(boxes) == 0:
                continue
            # Method below takes all partitions. For 9 boxes, there are 21 147 partitions,
            # so we try to limit the number of boxes. We achieve this by melting couples of boxes

            # Melt couples of boxes until there are 7 boxes left
            while len(boxes) > 7:
                # Take all combinations of 2 boxes
                couples = it.combinations(boxes, 2)
                scored_couples = []
                for c in couples:
                    new_loss = self.cover_loss(c)
                    original_loss = self.cover_loss([c[0]]) + self.cover_loss([c[1]])
                    ratio = new_loss / original_loss
                    # covered_area = cover_area(c)
                    # original_area = c[0]['width'] * c[0]['height'] + c[1]['width'] * c[1]['height']
                    # ratio = (covered_area / original_area) ** self.size_box_loss_power
                    scored_couples.append((ratio, c))
                scored_couples.sort(key=lambda x: x[0])
                best_couple = scored_couples[0][1]
                # Remove the two boxes and add the melted one
                boxes.remove(best_couple[0])
                boxes.remove(best_couple[1])
                boxes.append(self.melt_covering(best_couple))

            # Take all possible partitions of the remaining boxes
            scored_partitions = [(partition_loss(p), p) for p in partition(boxes)]
            scored_partitions.sort(key=lambda x: x[0])
            best_partition = scored_partitions[0][1]
            for covering in best_partition:
                melted_boxes.append(self.melt_covering(covering))
        return melted_boxes

    def remove_overlapping_boxes(self, predictions):
        def iou_box(box1, box2):
            """
            Calculate the IoU between two boxes
            """
            x1, y1, x2, y2 = box1['x'], box1['y'], box1['x'] + box1['width'], box1['y'] + box1['height']
            x3, y3, x4, y4 = box2['x'], box2['y'], box2['x'] + box2['width'], box2['y'] + box2['height']
            # Calculate IoU
            x5 = max(x1, x3)
            y5 = max(y1, y3)
            x6 = min(x2, x4)
            y6 = min(y2, y4)
            intersection = max(0, x6 - x5) * max(0, y6 - y5)

            A1 = (x2 - x1) * (y2 - y1)
            A2 = (x4 - x3) * (y4 - y3)
            if self.union_method_inter_class == 'union':
                u = A1 + A2 - intersection
            elif self.union_method_inter_class == 'min':
                u = min(A1, A2)
            elif self.union_method_inter_class == 'max':
                u = max(A1, A2)
            else:
                raise Exception(f'Unknown union method {self.union_method_inter_class}')
            iou = intersection / u
            return iou

        # Take all couples of boxes, if class is different and IoU > threshold, remove the one with lower confidence
        i = 0
        while i < len(predictions):
            # Take the first box
            pred = predictions[i]
            j = i + 1
            while j < len(predictions):
                # Take the second box
                other_pred = predictions[j]
                if pred['labels'] == other_pred['labels']:
                    j += 1
                    continue
                # Boxes are of different classes. Check IoU
                if iou_box(pred, other_pred) > self.iou_inter_class:
                    if self.overlap_behavior == 'remove':
                        # Remove the box with lower confidence
                        if pred['confidence'] > other_pred['confidence']:
                            predictions.pop(j)
                        else:
                            predictions.pop(i)
                            i -= 1
                            break
                    elif self.overlap_behavior == 'merge':
                        # Merge the boxes
                        predictions[i] = self.melt_covering((pred, other_pred))
                        predictions.pop(j)
                        # Restart the loop
                        i = -1
                        break
                    else:
                        raise Exception(f'Unknown overlap behavior {self.overlap_behavior}')
                j += 1
            i += 1
        return predictions

    def show_process_time(self):
        """ Show average processing time for each step """
        self.crono.print_stats()

    def get_BGR_mask(self):
        """Sum the damage masks and convert to BGR"""
        # Sum the damage masks
        color_idx = 1  # 0 for blue, 1 for green, 2 for red
        BGR_damage_mask = np.zeros((*self.damage_shape, 3), dtype=np.uint16)
        BGR_damage_mask[:, :, color_idx] = self.damage_mask_uint8.sum(axis=0)
        # Clip values above 255
        BGR_damage_mask[BGR_damage_mask > 255] = 255
        # Convert to uint8
        BGR_damage_mask = BGR_damage_mask.astype(np.uint8)
        # Scale up
        BGR_damage_mask = cv2.resize(BGR_damage_mask, (self.img_shape[1], self.img_shape[0]))

        return BGR_damage_mask

    def apply_homography(self, h):
        """
        Apply the homography to the damage mask

        Args:
            h (np.array): The homography matrix
        """
        for cls in range(self.n_classes):
            self.damage_mask[cls] = cv2.warpPerspective(self.damage_mask[cls], h, self.damage_shape)

    def reset_damage_mask(self):
        self.damage_mask = self.get_empty_mask()

    def get_new_damage_mask(self, predictions):
        """
        Generate an empty damage mask and draw the predictions on it
        """
        new_damage_mask = self.get_empty_mask(1)
        # Sort predictions by increasing confidence. This way, the boxes with the highest confidence will be drawn last
        # The other way around, the boxes with the lowest confidence would be drawn last and could overwrite the other boxes
        # This solution is computationally less expensive than apply nm.minimum on each box assingment
        predictions = sorted(predictions, key=lambda k: k['confidence'])

        for pred in predictions:
            # Retrieve the coordinates of the box
            x = int(pred['x'] * self.scale_mask_factor)
            y = int(pred['y'] * self.scale_mask_factor)
            w = int(abs(int(pred['width'])) * self.scale_mask_factor)
            h = int(abs(int(pred['height'])) * self.scale_mask_factor)
            conf = pred['confidence']
            cls = self.dict_labels[pred['labels']]

            # Draw the box on the new damage mask
            new_damage_mask[cls, y:y + h, x:x + w] = 1 - conf

        return new_damage_mask

    def get_empty_mask(self, value=0):
        """Generate an empty damage mask"""
        if value == 0:
            return np.zeros([self.n_classes, *self.damage_shape], dtype=np.float32)
        else:
            return np.ones([self.n_classes, *self.damage_shape], dtype=np.float32) * value


# ╦ ╦╔═╗╦  ╔═╗╔═╗╦═╗  ╔╦╗╔═╗╔╦╗╦ ╦╔═╗╔╦╗╔═╗
# ╠═╣║╣ ║  ╠═╝║╣ ╠╦╝  ║║║║╣  ║ ╠═╣║ ║ ║║╚═╗
# ╩ ╩╚═╝╩═╝╩  ╚═╝╩╚═  ╩ ╩╚═╝ ╩ ╩ ╩╚═╝═╩╝╚═╝
# -------------------------------------------------
def draw_boxes(frame, predictions):
    "Draw prediction boxes on frame"
    new_frame = frame.copy()
    for box in predictions:
        x = int(box['x'])
        y = int(box['y'])
        w = abs(int(box['width']))
        h = abs(int(box['height']))
        # confidence
        conf = box['confidence']
        label = box['labels']

        # draw the box
        cv2.rectangle(new_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # draw the confidence (2 digits) and the label (7 chars)
        cv2.putText(new_frame, f'{conf:.2f} {label[:7]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

    return new_frame
