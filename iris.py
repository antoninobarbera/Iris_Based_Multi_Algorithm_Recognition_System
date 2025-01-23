import cv2 as cv
import numpy as np
from tools.segmentation import *
from tools.utils import point_in_circle, draw_keypoints_image, warp_polar, encoding


class iris_class:
    
    __slots__ = ['image', 'config', 'centre', 'idx',
                'pupil_radius', 'iris_radius', 'segmented',
                'keypoints', 'descriptors', 'keypoints_image', 'normalize', 'iris_code']

    def __init__(self, image, idx, config):
        """
        Initializes the iris_class object.

        Parameters:
        - image: Input image containing the eye.
        - idx: Index or identifier for the iris sample.
        - config: Configuration object containing various parameters.
        """
        self.image = image
        self.config = config
        self.idx = idx

    def segmentation(self):
        """
        Segments the iris region from the input image.

        Steps:
        1. Converts the input image to grayscale.
        2. Detects the pupil and iris boundaries using configuration parameters.
        3. Pads the image for processing.
        4. Removes the pupil region and equalizes the remaining iris section.
        5. Crops the image to the size of the iris region.
        
        Result:
        - The segmented iris is saved in the `self.segmented` attribute.
        """
        gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        centre, self.pupil_radius = detect_pupil(gray, self.config)
        _, self.iris_radius = detect_iris(gray, self.config)
        padded_image, self.centre = padding(gray, centre, self.config.segmentation.padded_x_size, self.config.segmentation.padded_y_size)
        circle = section_circle_area(padded_image, self.centre, self.iris_radius)
        iris_section = remove_pupil(circle, self.centre, self.pupil_radius)
        equalized_image = equalize(iris_section, self.config)
        cropped_image, self.centre = crop_padding(equalized_image, self.iris_radius, self.config)
        self.segmented = cropped_image

    def normalization(self):
        """
        Normalizes the segmented iris region into polar coordinates.

        Steps:
        1. Warps the segmented iris image into polar coordinates.
        2. Inverts pixel values and converts the result to an 8-bit image.
        3. Crops the normalized image using the configuration border size.
        
        Result:
        - The normalized iris image is saved in the `self.normalize` attribute.
        """
        image = self.get_segmented_image()
        x_size = self.config.normalization.x_size
        y_size = self.config.normalization.y_size        
        polar_iris = warp_polar(image, x_size, y_size, self.centre, self.pupil_radius, self.iris_radius)
        polar_iris_inv = 255 - polar_iris
        polar_iris_uint8 = polar_iris_inv.astype(np.uint8)
        normalized_iris = polar_iris_uint8[0:self.config.normalization.border, :]
        self.normalize = normalized_iris
         
    def feature_extraction(self):
        """
        Extracts keypoints and descriptors from the segmented iris using SIFT.

        Steps:
        1. Detects keypoints using SIFT.
        2. Filters valid keypoints based on their position:
           - Inside the iris region.
           - Outside the pupil region.
        3. Computes descriptors for the valid keypoints.
        4. Draws keypoints on the segmented image for visualization.

        Results:
        - The keypoints are saved in `self.keypoints`.
        - The descriptors are saved in `self.descriptors`.
        - The visualization image is saved in `self.keypoints_image`.
        """
        sift = cv.SIFT_create()
        kp_found = sift.detect(self.segmented, None)
        valid_kp = []
        for kp in kp_found:
            point = (kp.pt[0], kp.pt[1])
            kp_in_iris = point_in_circle(self.centre, self.iris_radius, point)
            kp_out_pupil = not point_in_circle(self.centre, self.pupil_radius, point)
            if kp_in_iris and kp_out_pupil:
                valid_kp.append(kp)
        self.keypoints, self.descriptors = sift.compute(self.segmented, valid_kp)
        self.keypoints_image = draw_keypoints_image(self.segmented, valid_kp, self.centre, self.iris_radius)

    def set_iris_code(self):
        """
        Encodes the normalized iris into a binary code.

        Steps:
        1. Calls the `normalization` method to ensure the iris is normalized.
        2. Uses the `encoding` function to generate a binary iris code.

        Result:
        - The binary iris code is saved in the `self.iris_code` attribute.
        """
        self.normalization()
        self.iris_code = encoding(self.normalize, self.config)

    def get_idx(self):
        """
        Returns the index or identifier of the iris sample.
        """
        return self.idx
    
    def get_image(self):
        """
        Returns the original input image.
        """
        return self.image
 
    def get_iris_code(self):
        """
        Returns the encoded binary iris code.
        """
        return self.iris_code
    
    def get_keypoints(self):
        """
        Returns the detected keypoints from the segmented iris.
        """
        return self.keypoints
    
    def get_descriptors(self):
        """
        Returns the descriptors computed for the valid keypoints.
        """
        return self.descriptors
    
    def get_segmented_image(self):
        """
        Returns the segmented iris image.
        """
        return self.segmented
    
    def get_keypoints_image(self):
        """
        Returns the image with keypoints drawn for visualization.
        """
        return self.keypoints_image
    
    def get_normalized_image(self):
        """
        Returns the normalized iris image in polar coordinates.
        """
        return self.normalize
    
    def get_attributes(self):
        """
        Returns key attributes of the iris:
        - Centre of the iris.
        - Pupil radius.
        - Iris radius.
        """
        return self.centre, self.pupil_radius, self.iris_radius
    