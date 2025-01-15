import cv2 as cv
import numpy as np
from tools.segmentation import *
from tools.utils import point_in_circle, draw_keypoints_image, warp_polar, encoding


class iris_class:
    __slots__ = ['image', 'config', 'centre', 'idx',
                'pupil_radius', 'iris_radius', 'segmented',
                'keypoints', 'descriptors', 'keypoints_image', 'normalize', 'iris_code']

    def __init__(self, image, idx, config):
       self.image = image
       self.config = config
       self.idx = idx


    def segmentation(self):
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
        image = self.get_segmented_image()
        x_size = self.config.normalization.x_size
        y_size = self.config.normalization.y_size        
        polar_iris = warp_polar(image, x_size, y_size, self.centre, self.pupil_radius, self.iris_radius)
        polar_iris_inv = 255 - polar_iris
        polar_iris_uint8 = polar_iris_inv.astype(np.uint8)
        normalized_iris = polar_iris_uint8[0:self.config.normalization.border, :]
        self.normalize = normalized_iris
         
    def feature_extraction(self):
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
        self.normalization()
        self.iris_code = encoding(self.normalize, self.config)

    def get_idx(self):
        return self.idx
 
    def get_iris_code(self):
        return self.iris_code
    
    def get_keypoints(self):
        return self.keypoints
    
    def get_descriptors(self):
        return self.descriptors
    
    def get_segmented_image(self):
        return self.segmented
    
    def get_keypoints_image(self):
        return self.keypoints_image
    
    def get_attributes(self):
        return self.centre, self.pupil_radius, self.iris_radius
    
    