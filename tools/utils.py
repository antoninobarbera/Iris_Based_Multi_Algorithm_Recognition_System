import math
import cv2 as cv
import numpy as np


def distance(point_1, point_2):
    x1, y1 = point_1
    x2, y2 = point_2
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

def point_in_circle(centre, radius, point):
    dist = distance(centre, point) 
    return dist <= radius

def draw_keypoints_image(image_iris, keypoints, centre, iris_radius):
    red = (0, 0, 255)
    blue = (255, 0, 0)
    keypoints_image = cv.cvtColor(image_iris.copy(), cv.COLOR_GRAY2BGR)
    cv.circle(keypoints_image, centre, 2, red, thickness=3)
    cv.circle(keypoints_image, centre, iris_radius + 3, red, thickness=3)
    keypoints_image = cv.drawKeypoints(keypoints_image, keypoints, color=blue, flags=0, outImage=None)
    return keypoints_image

def angle(point_1, point_2):
    x1, y1 = point_1
    x2, y2 = point_2
    angle = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
    angle_360 = (angle + 360) % 360
    return angle_360

def to_polar(point, pole=(0, 0)):
    r = distance(point, pole)
    theta = angle(point, pole)
    return r, theta

def normalize_r(r, pupil_radius, iris_radius):
    range = iris_radius - pupil_radius
    r_norm = (r - pupil_radius) / range
    return r_norm

def is_within_one_std(value, centre, dev):
    not_in_left_tail = value > centre - dev
    not_in_right_tail = value < centre + dev
    is_on_range = not_in_left_tail and not_in_right_tail
    return is_on_range

def warp_polar(image, x_size, y_size, centre, pupil_radius, iris_radius):
    normalized_iris=np.zeros(shape=(x_size, y_size))
    x_c, y_c = centre
        
    angle= 2.0 * math.pi / y_size
    inner_boundary_x = np.zeros(shape=(1, y_size))
    inner_boundary_y = np.zeros(shape=(1, y_size))
    outer_boundary_x = np.zeros(shape=(1, y_size))
    outer_boundary_y = np.zeros(shape=(1, y_size))

    for i in range(y_size):
        inner_boundary_x[0][i]= x_c + pupil_radius * math.cos(angle*(i))
        inner_boundary_y[0][i]= y_c + pupil_radius * math.sin(angle*(i))
        outer_boundary_x[0][i]= x_c + iris_radius * math.cos(angle*(i))
        outer_boundary_y[0][i]= y_c + iris_radius * math.sin(angle*(i))
        
    for j in range (y_size):
       for i in range (x_size):
            normalized_iris[i][j]= image[min(int(int(inner_boundary_y[0][j]) + (int(outer_boundary_y[0][j]) - int(inner_boundary_y[0][j])) * (i/64.0)),
                                        image.shape[0] - 1)][min(int(int(inner_boundary_x[0][j]) + (int(outer_boundary_x[0][j]) - int(inner_boundary_x[0][j])) * (i/64.0)),
                                        image.shape[1] - 1)]
            
    return normalized_iris

def gabor_filter(theta, config):
    ksize = (config.gabor_filter.x_size, config.gabor_filter.y_size)
    kernel = cv.getGaborKernel(ksize, 
                                config.gabor_filter.gamma, 
                                theta,
                                config.gabor_filter.frequency,
                                config.gabor_filter.sigma,
                                config.gabor_filter.psi, 
                                ktype=cv.CV_64F)
    return kernel

def encoding(image, config):
    vector = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]:
        gabor = gabor_filter(theta, config)
        filtered_eye = cv.filter2D(image, cv.CV_64F, gabor)
        for i in range(0, image.shape[0], 8):
            for j in range(0, image.shape[1], 8):                
                patch = filtered_eye[i:i+8, j:j+8]
                mean = patch.mean()
                AAD = np.abs(patch - mean).mean()
                vector.extend([mean, AAD]) 
    return np.array(vector)

def manage_best_model_and_metrics(model, evaluation_metric, metrics, best_metric, best_model, lower_is_better):
    if lower_is_better:
        is_best = metrics[evaluation_metric] < best_metric
    else:
        is_best = metrics[evaluation_metric] > best_metric
        
    if is_best:
        best_metric = metrics[evaluation_metric]
        best_model = model

    return best_model, best_metric
        