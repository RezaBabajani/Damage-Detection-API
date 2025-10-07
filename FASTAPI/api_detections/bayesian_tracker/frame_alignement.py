from typing import Optional
import cv2
import numpy as np
import math
from unittest.mock import Mock
import jsonpickle


hom_cov = np.array([[0.00861490115070552, -0.008104841397013734, 1.3561676716411288, 0.01639377782186947, -0.019404331157951794, 2.8124147058597098, 9.4025673378114e-05, -5.31185774935879e-05], [-0.008104841397013734, 0.01411756987664459, -3.3257454986673416, -0.0233614344152331, 0.033736091890959885, -5.324036940680971, -0.00013896868863756622, 8.573893297934093e-05], [1.3561676716411288, -3.3257454986673416, 1140.8160020256564, 5.462960276381201, -8.448754967407552, 1397.2255685519863, 0.033384712961459694, -0.020596367311927887], [0.01639377782186947, -0.0233614344152331, 5.462960276381201, 0.04308010178221681, -0.05893624929664737, 9.237591611810677, 0.00025429818827526996, -0.000151120565870334], [-0.019404331157951794, 0.033736091890959885, -8.448754967407552, -0.05893624929664737, 0.08576649255918471, -13.798947949692657, -0.0003532706413357414, 0.00021528082707609927],
                [2.8124147058597098, -5.324036940680971, 1397.2255685519863, 9.237591611810677, -13.798947949692657, 2370.3139296115883, 0.05624421026498169, -0.034265240746514394], [9.4025673378114e-05, -0.00013896868863756622, 0.033384712961459694, 0.00025429818827526996, -0.0003532706413357414, 0.05624421026498169, 1.51765208126364e-06, -9.03302901525101e-07], [-5.31185774935879e-05, 8.573893297934093e-05, -0.020596367311927887, -0.000151120565870334, 0.00021528082707609927, -0.034265240746514394, -9.03302901525101e-07, 5.472627903599845e-07]])
delta_h_cov_matrix = np.array([[61.42517754620388, 33.86528302659483, -5586.2103292142865, 46.37110654656019, 56.445618655876565, -6468.629137076408, 0.4189984150939403, -0.029372706421444454, 0.0], [33.86528302659483, 61.10544168573443, -6220.243209207191, 18.247515611044314, 43.020999792356996, -4161.13546216157, 0.18454592675959142, 0.07796491270569464, 0.0], [-5586.2103292142865, -6220.243209207191, 968324.1803299667, -2600.9862288963527, -6987.217795002534, 963623.679168421, -28.166508401850912, -6.047576443373201, 0.0], [46.37110654656019, 18.247515611044314, -2600.9862288963527, 48.45929641789468, 27.655249762306767, -2000.9584921662931, 0.36323899789297326, -0.05451513338810013, 0.0], [56.445618655876565, 43.020999792356996, -6987.217795002534, 27.655249762306767, 70.22549996824947, -9080.93015168046, 0.3371938670818854, 0.016155701799315705, 0.0],
                                   [-6468.629137076408, -4161.13546216157, 963623.679168421, -2000.9584921662931, -9080.93015168046, 1452049.2729229836, -32.55193136241706, -2.4455958696297326, 0.0], [0.4189984150939403, 0.18454592675959142, -28.166508401850912, 0.36323899789297326, 0.3371938670818854, -32.55193136241706, 0.0031205581480410337, -0.00037240281590683355, 0.0], [-0.029372706421444454, 0.07796491270569464, -6.047576443373201, -0.05451513338810013, 0.016155701799315705, -2.4455958696297326, -0.00037240281590683355, 0.00026181811758531446, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

hom_mean = np.eye(3).flatten()[:-1]
hom_cov_inv = np.linalg.inv(hom_cov)



score_functions = dict()
# Create a decorator for registering score functions
def score_function(func):
    score_functions[func.__name__] = func
    return func

@score_function
def distance_damage_v0(distance, damage):
    """
    Score a match point based on damage and distance
    """
    return 1 / (distance +0.01) * (damage + 10)

@score_function
def distance_damage_v1(distance, damage, alfa=0.25):
    """
    Score a match point based on damage and distance
    """
    return (damage/125) * alfa + (-distance/50) * (1-alfa)


@score_function
def distance_only(distance, damage):
    """
    Score a match point based on damage and distance
    """
    return - distance

class Match:
    """
    Class representing a match between two images
    """
    def __init__(self, distance=None, damage=None, prev_point=None, next_point=None):
        self.distance = distance
        self.damage = damage
        self.prev_point = prev_point
        self.next_point = next_point
        self.list = None
        self._score = None

    @property
    def score(self):
        if self._score:
            return self._score
        if self.list is not None:
            self._score = self.list.score_function(self.distance, self.damage, **self.list.score_function_kwargs)
            return self._score
        else:
            raise Exception("Match not added to a list")



class ListMatches(list):
    """
    Class representing a list of matches between two images
    """
    def __init__(self, score_function, **kwargs):
        # Call the base class constructor
        super().__init__()
        # Add the score function
        self.score_function = score_function
        self.score_function_kwargs = kwargs

    def append(self, match):
        # Call the base class append method
        super().append(match)
        match.list = self

    def sort(self):
        super().sort(key=lambda match: match.score, reverse=True)

    def clear(self) -> None:
        # Just to remember that this method exists and you have to use it
        super().clear()

    def draw_match_points(self, frame, colormap='jet'):
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import matplotlib.cm as cmx
        """
        Draw the features points on the frame, using a color gradient to indicate the score

        Parameters
        ----------
        frame : np.array
            The frame to draw on

        Returns
        -------
        np.array
            The frame with the features points drawn on it
        """

        if len(self) == 0:
            return frame
        frame = frame.copy()
        # Get min and max scores
        # Note: Usually matches are sorted by score, so the first and last match have the min and max score
        #       But just in case, we calculate it
        min_score = min(self, key=lambda match: match.score).score
        max_score = max(self, key=lambda match: match.score).score
        # Choose a colormap
        cmap = plt.get_cmap(colormap)
        # Create a normalization function
        norm = colors.Normalize(vmin=min_score, vmax=max_score)
        # Create a ScalarMappable object
        scalar_map = cmx.ScalarMappable(norm=norm, cmap=cmap)

        for match in self:
            color = scalar_map.to_rgba(match.score)
            bg_color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))

            # Draw the point whith a white outline
            cv2.circle(frame, match.next_point, 5, (255, 255, 255), 1)
            cv2.circle(frame, match.next_point, 4, bg_color, -1)
        # Add the gradient color bar
        color_bar = np.zeros((100, 20, 3), dtype=np.uint8)
        for i in range(100):
            color = cmap(i/100)
            color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
            cv2.line(color_bar, (0, i), (20, i), color, 1)
        frame[0:100, 0:20] = color_bar

        return frame



class KalmanFiltering:
    """Implementation of a Kalman filter for the homography matrix

    Our state is represented by the 9 parameters of the homography matrix
    hcov_matrix

    """
    # Matrix below is the result of measuring on 4000 frames over 9 videos
    def __init__(self, ratio_noise=0.2):
        """
        ratio_noise: float in [0, 1] that determines the proportion of the noise that is measurement noise (the rest is process noise)
                     An higher value make the filter more conservative and less reactive to new measurements
        """
        # Measurmement noise
        self.R = delta_h_cov_matrix * np.sqrt(ratio_noise)
        # Process noise
        self.Q = delta_h_cov_matrix * np.sqrt(1 - ratio_noise)
        self.reset()


    def reset(self):
        """Reset the filter to its initial state"""
        # Starting state is the identity homography matrix
        self.x_n = np.eye(3).flatten()
        # Starting priori error is big
        self.P_n = np.eye(9) * 1000
        # Current homography matrix is not valid
        self.x_is_valid = False


    def update(self, z_n):
        """Update the filter with a new measurement"""

        # If no measurement is given, reset the filter
        if z_n is None:
            self.reset()
            return
        # z_n is the new measurement (a new homography matrix)
        # First linearize the homography matrix
        z_n = z_n.flatten()
        # Priori estimate (no action matrix)
        self.x_p = self.x_n
        # Add process error to priori error
        self.P_p = self.P_n + self.Q
        # Kalman gain
        self.K = self.P_p @ np.linalg.pinv(self.P_p + self.R)
        # Posteriori estimate
        self.x_n = self.x_p + self.K @ (z_n - self.x_p)
        # Posteriori error
        self.P_n = (np.eye(9) - self.K) @ self.P_p
        # Current homography matrix is valid
        self.x_is_valid = True

    def last_state(self):
        """Returns the last homography matrix"""
        if not self.x_is_valid:
            return None
        return self.x_n.reshape((3, 3))


class FrameAlignement:

    def __init__(
            self,
            tracker: 'BayesianTracker', # Tracker object calling this class
            ratio_noise: float=0.2, # Kalman filter parameter
            use_kalman: bool=False, # Whether to use Kalman filter
            max_features: int=400, # Maximum number of features to detect
            good_match_percent: float=0.5, # Percentage of good matches to keep
            min_num_matches: int=10, # Minimum number of matches to accept homography
            img_shape: tuple=(640,640), # Shape of the image
            score_features_method: str='distance_damage_v1', # Method to score features
            score_function_kwargs: dict={}, # Additional arguments for the score function
            feature_extractor: str='ORB', # feature_extractor algorithm to use
            matching_method: str='bf-knn', # Matching method to use for corresponding keypoints (bf-hamming, flann, brute-force)
            norm_method: str='displacement', # Norm to use for the homography matrix. Acceptable values are 'mahalanobis', 'iou', 'displacement'
            max_h_norm: float=0.2, # Maximum acceptable norm of the homography matrix
            crono: Optional['Chronometer']=None, # Whether to print the time taken by each step
    ):
        self.MAX_FEATURES = max_features
        self.GOOD_MATCH_PERCENT = good_match_percent
        self.use_kalman = use_kalman
        self.min_num_matches = min_num_matches
        self.img_shape = img_shape
        # Initialize coordinates of a fixed point as numpy vector at center of image
        self.fixed_point = np.array([100, 100, 1])
        self.base_frame = None
        self.h = None
        self.tracker = tracker
        self.ratio_noise = ratio_noise
        self.max_h_norm = max_h_norm

        self.h_mean_mask = 0

        # Choose the score function
        self.score_function = score_functions.get(score_features_method, None)
        if self.score_function is None:
            raise ValueError(f'Unknown score_features_method: {score_features_method}')

        # List of match points with scores (for easy of implementation and plotting)
        self.matches = ListMatches(score_function=self.score_function, **score_function_kwargs)
        # List of detected features (for caching)
        self.prev_keypoints = None
        self.prev_descriptors = None

        if feature_extractor == 'ORB':
            self.feature_extractor = cv2.ORB_create(self.MAX_FEATURES)
        elif feature_extractor == 'BRISK':
            self.feature_extractor = cv2.BRISK_create(self.MAX_FEATURES)
        elif feature_extractor == 'AKAZE':
            self.feature_extractor = cv2.AKAZE_create(self.MAX_FEATURES)
        elif feature_extractor == 'KAZE':
            self.feature_extractor = cv2.KAZE_create(self.MAX_FEATURES)
        elif feature_extractor == 'SIFT':
            self.feature_extractor = cv2.SIFT_create(self.MAX_FEATURES)
        else:
            raise ValueError(f'Unknown matching_algorithm: {feature_extractor}')

        if matching_method == 'brute-force':
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            self._match = lambda feat1, feat2: self.matcher.match(feat1, feat2)
        elif matching_method == 'flann':
            FLANN_INDEX_LSH = 6
            # Change parameters below if SIFT instead of ORB
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,
                                key_size=12,
                                multi_probe_level=1)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            self._match = lambda feat1, feat2: self.matcher.knnMatch(feat1, feat2, k=2)
        elif matching_method == 'bf-knn':
            self.matcher = cv2.BFMatcher()
            self._match = lambda feat1, feat2: self.matcher.knnMatch(feat1, feat2, k=2)
        elif matching_method == 'bf-hamming':
            self.matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
            self._match = lambda feat1, feat2: self.matcher.match(feat1, feat2)

        # Initialise ORB matcher using fake image
        fake_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2GRAY)
        self.feature_extractor.detectAndCompute(fake_img, None)


        self.draw_match_points = self.matches.draw_match_points
        self.kalman = KalmanFiltering(ratio_noise=self.ratio_noise)

        # Choose the norm method
        if norm_method == 'mahalanobis':
            self.norm = self.mahalanobis_norm
        elif norm_method == 'iou':
            self.norm = self.IoU_h_matrix
        elif norm_method == 'displacement':
            self.norm = self.displacement_norm
        else:
            raise ValueError(f'Unknown norm_method: {norm_method}')

        # If no crono is provided, use a mock object (i.e. do nothing)
        if crono is None:
            self.crono = Mock()
        else:
            self.crono = crono

        self.h_norm = math.inf

    def match(self, feat1, feat2):
        matches = self._match(feat1, feat2)
        if not matches:
            return []
        # Check matches[0] is iterable usint type
        if type(matches[0]) not in (tuple, list):
            return matches
        best_matches = []
        for m in matches:
            if len(m) == 0:
                continue
            elif len(m) == 1:
                best_matches.append(m[0])
            else:
                if m[1].distance < 0.75 * m[0].distance:
                    best_matches.append(m[1])
                else:
                    best_matches.append(m[0])
        return best_matches



    def process(self, new_frame):
        """
        Process a new frame and return True if no error occured
        """
        msg = ""

        # Only for the first image let's initialize the base_frame (i.e. previous frame)
        # No need to calculate the homography in this case. More specifically, we need to not do it
        # since this generate an error (matchpoints are found but there is no previous keypoint)
        if self.base_frame is None:
            self.base_frame = new_frame.copy()
            h = None
        else:
            h, mask = self.find_homography(self.base_frame, new_frame.copy())

        if h is None:
            msg = "Impossible to find homography"

        self.crono.start("in_IoU_h_matrix")
        # Calculate how much the homography matrix change the image using IoU
        # (value is little for big changes, 0 everything change, 1 nothing change)
        # iou_h = self.IoU_h_matrix(h)
        # print("IoU_h: ", iou_h)
        # if iou_h < self.MIN_IOU_FOR_HOMOGRAPHY:
        #     msg = "IoU_h too low: " + str(iou_h)
        #     h = None

        # Calculate how much the homography matrix change the image using average displacement
        self.h_norm = self.norm(h)
        print("h_norm: ", self.h_norm)
        if self.h_norm > self.max_h_norm:
            msg = "h_norm too high: " + str(self.h_norm)
            h = None


        self.crono.stop("in_IoU_h_matrix")

        self.crono.start("kalman")
        # Apply Kalman filter
        if self.use_kalman:
            self.kalman.update(h)
            h = self.kalman.last_state()
        self.crono.stop("kalman")


        self.h = h
        self.base_frame = new_frame

        success = self.h is not None
        return success, msg

    def export_state(self):
        """
        Save the current state of the tracker
        """
        if self.prev_keypoints is None:
            prev_keypoints_tuple = None
        else:
            prev_keypoints_tuple = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in self.prev_keypoints]
        prev_keypoints_json = jsonpickle.dumps(prev_keypoints_tuple)

        current_state = {
            'base_frame': jsonpickle.dumps(self.base_frame),
            'h': jsonpickle.dumps(self.h),
            'prev_keypoints': prev_keypoints_json,
            'prev_descriptors': jsonpickle.dumps(self.prev_descriptors),
            'kalman': jsonpickle.dumps(self.kalman),
            }
        return current_state

    def load_state(self, state):
        """
        Load a previous state of the tracker
        """
        self.base_frame = jsonpickle.loads(state['base_frame'])
        self.h = jsonpickle.loads(state['h'])
        prev_keypoints_tuple = jsonpickle.loads(state['prev_keypoints'])
        if prev_keypoints_tuple is None:
            self.prev_keypoints = None
        else:
            self.prev_keypoints = [cv2.KeyPoint(x=p[0][0], y=p[0][1], size=p[1], angle=p[2], response=p[3], octave=p[4], class_id=p[5]) for p in prev_keypoints_tuple]
        self.prev_descriptors = jsonpickle.loads(state['prev_descriptors'])
        self.kalman = jsonpickle.loads(state['kalman'])


    def reset_state(self):
        """
        Reset the tracker to its initial state
        """
        self.base_frame = None
        self.h = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.kalman = KalmanFiltering(ratio_noise=self.ratio_noise)

    def find_homography(self, im1, im2):

        self.crono.start("pr_find_features")
        if self.prev_keypoints is None:
            im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            self.prev_keypoints, self.prev_descriptors = self.feature_extractor.detectAndCompute(im1Gray, None)
        self.crono.stop("pr_find_features")


        self.crono.start("find_features")
        # Convert images to grayscale
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        # Detect ORB features and compute descriptors.
        keypoints2, descriptors2 = self.feature_extractor.detectAndCompute(im2Gray, None)
        self.crono.stop("find_features")

        # Match features.
        self.crono.start("match_features")
        try:
            matches = self.match(self.prev_descriptors, descriptors2)
        except cv2.error:

            self.crono.stop("match_features")
            print("Error in find_homography, no match found")
            # Save keypoints and descriptors for next frame
            self.prev_keypoints = keypoints2
            self.prev_descriptors = descriptors2
            return None, None


        self.crono.stop("match_features")

        self.crono.start("calc_homography")

        # Filter using the damage mask
        self.crono.start("score_points")

        # Get the mask, with good size

        if self.tracker.damage_mask_uint8 is not None:
            mask = self.tracker.damage_mask_uint8.sum(axis=2, dtype=np.uint16)
            mask.clip(0, 255, out=mask)
            # Resize to img shape
            mask = cv2.resize(mask, (im2.shape[1], im2.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Maybe define a special class for matches and list of matches?

            # Evaluate mask on all features points found (check previous frame only)
            self.matches.clear()
            for i, raw_match in enumerate(matches):
                prev_p = self.prev_keypoints[raw_match.queryIdx].pt
                next_p = keypoints2[raw_match.trainIdx].pt

                match = Match(prev_point=(int(prev_p[0]), int(prev_p[1])),
                              next_point=(int(next_p[0]), int(next_p[1])),
                              distance=raw_match.distance)
                match.damage = mask[match.prev_point[1], match.prev_point[0]]
                self.matches.append(match)


            # Sort matches by score
            self.matches.sort()
        self.crono.stop("score_points")
        # Save keypoints and descriptors for next frame
        self.prev_keypoints = keypoints2
        self.prev_descriptors = descriptors2

        # Remove not so good matches
        numGoodMatches = int(len(self.matches) * self.GOOD_MATCH_PERCENT)
        good_matches = self.matches[:numGoodMatches]

        n_matches = len(good_matches)
        print(f"Found {n_matches} matches")
        if n_matches < self.min_num_matches:
            print(f"Found only {n_matches} matches, not enough ({self.min_num_matches} required)")
            self.crono.stop("calc_homography")
            return None, None


        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(good_matches):
            points1[i, :] = match.prev_point
            points2[i, :] = match.next_point

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        self.h_mean_mask = (sum(mask) / len(mask))[0]
        self.crono.stop("calc_homography")



        return h, mask

    def IoU_h_matrix(self, h):
        """Calculate the IoU between the square [0,1]x[0,1] and its image through the homography matrix
        """

        def polygon_iou(polygon1, polygon2):
            # Convert the polygons to OpenCV contours
            contour1 = np.array(polygon1, dtype=np.float32).reshape((-1, 1, 2))
            contour2 = np.array(polygon2, dtype=np.float32).reshape((-1, 1, 2))

            # Find the intersection of the two contours
            _, intersection = cv2.intersectConvexConvex(contour1, contour2)

            # Calculate the area of the intersection
            inter_area = 0 if intersection is None else cv2.contourArea(intersection)

            # Calculate the area of each polygon
            area1 = cv2.contourArea(contour1)
            area2 = cv2.contourArea(contour2)

            # Calculate the union of the two polygons
            union_area = area1 + area2 - inter_area

            # Calculate the IOU
            iou = inter_area / union_area
            return 1 - iou
        if h is None:
            return np.nan

        # Create the square
        square = np.array([[0, 0, 1], [self.img_shape[1], 0, 1], [self.img_shape[1], self.img_shape[0], 1], [0, self.img_shape[0], 1]]).T
        # Apply the homography
        new_square = h @ square
        new_square = new_square / new_square[2, :]
        # Calculate the IoU
        return polygon_iou(square[:2, :].T, new_square[:2, :].T)

    def displacement_norm(self, h):
        """
        Calculate average displacement of the four corners of the image divided by the image mean side
        """
        if h is None:
            return np.nan
        # Normalize the homography matrix
        h = h / h[2, 2]

        points = np.array([[0, 0, 1],
                           [self.img_shape[0] - 1, 0, 1],
                           [self.img_shape[0] - 1, self.img_shape[1] - 1, 1],
                           [0, self.img_shape[1] - 1, 1]], dtype=np.float32).T

        # Apply the homography matrix to the points
        transformed_points = h @ points
        transformed_points = transformed_points / transformed_points[2, :]

        # Calculate the Euclidean distance between the original and transformed points
        distances = np.linalg.norm(points[:2, :] - transformed_points[:2, :], axis=0)

        # Compute the norm (e.g., average or maximum) of the displacements
        avg_displacement = np.mean(distances)
        # max_displacement = np.max(distances)
        mean_shape = np.mean(self.img_shape)
        displ_norm = avg_displacement / mean_shape
        print("-----")
        print(f"H norm: {displ_norm}\n H: {h}")
        print(f"H mask: {self.h_mean_mask}")
        return displ_norm


    def mahalanobis_norm(self, h):
        """
        Calculate the Mahalanobis distance of the homography matrix from the mean homography matrix
        Covariance matrix is given by extracting homography matrix from over 10 000 frames in 18 videos
        Attention!!: covariance matrix depends on frame size. You need to rescale everything if you change frame size
                     This is a base change, refer to BayesianTracker.h method for an example
        """
        # Normalize the homography matrix (generally it is already normalized)
        if h is None:
            return np.nan
        h = h / h[2, 2]
        h_vector = h.flatten()[:-1]
        # Calculate the Mahalanobis distance
        mahal_dist = np.sqrt((h_vector - hom_mean) @ hom_cov_inv @ (h_vector - hom_mean))
        # Multiply by 0.01 to have a value not too different from displacement norm
        # Actually, mean values for two methods become really similar using this factor
        # This is totally arbitrary
        return mahal_dist * 0.01