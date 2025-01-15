import cv2 as cv
import numpy as np

def detect_pupil(image, config):
    config = config.segmentation.pupil
    blurred_gray = cv.medianBlur(image, 5)
    img_canny = cv.Canny(blurred_gray, threshold1=config.canny_threshold_1, threshold2=config.canny_threshold_2)
    circles = cv.HoughCircles(
                        img_canny,
                        cv.HOUGH_GRADIENT, 
                        dp=config.dp, 
                        minDist=config.minDist, 
                        param1=config.param1,
                        param2=config.param2, 
                        minRadius=config.minRadius,
                        maxRadius=config.maxRadius
    )
    if circles is not None:
       circles = np.round(circles[0, :]).astype("int")
       x, y, r = circles[0]
       return (x, y), r
    else:
       print(' No Pupil Detected')
       return image, None
    
def detect_iris(image, config):
   config = config.segmentation.iris
   equalized = cv.equalizeHist(image)
   blurred_equalize = cv.medianBlur(equalized, 5)
   _, thresh = cv.threshold(blurred_equalize, 127, 255, cv.THRESH_BINARY)
   img_canny = cv.Canny(thresh, threshold1=config.canny_threshold_1, threshold2=config.canny_threshold_2)
   circles = cv.HoughCircles(img_canny,
                        cv.HOUGH_GRADIENT, 
                        dp=config.dp, 
                        minDist=config.minDist, 
                        param1=config.param1,
                        param2=config.param2, 
                        minRadius=config.minRadius,
                        maxRadius=config.maxRadius)
   if circles is not None:
       circles = np.round(circles[0, :]).astype("int")
       x, y, r = circles[0]
       return (x, y), r
   else:
       print(' No Iris Detected')
       return image, None

def remove_pupil(image, pupil_centre, pupil_radius):
    mask = 255 - (np.zeros_like(image))
    cv.circle(mask, pupil_centre, pupil_radius, (0, 0, 0), -1)
    image_without_pupil = cv.bitwise_and(image, mask)
    return image_without_pupil

def section_circle_area(image, centre, radius):
    mask = np.zeros_like(image) 
    cv.circle(mask, centre, radius, (255, 255, 255), -1)
    circle_area = cv.bitwise_and(image, mask)
    return circle_area

def padding(image, centre, x_size, y_size):
    if x_size % 2 != 0 :
        x_size += 1
    if y_size % 2 != 0 :
        y_size += 1

    x_size /= 2
    y_size /= 2
    xc, yc = centre
    h, w = image.shape[:2] 
    left_pad = int(x_size - xc)
    right_pad = int(x_size - (w - xc))
    top_pad = int(y_size - yc)
    bottom_pad = int(y_size - (h - yc))
    padded_image = cv.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv.BORDER_CONSTANT, value=0)
    centre = (int(x_size), int(y_size))
    return padded_image, centre

def equalize(image, config):
    config = config.segmentation
    equalized_image = cv.equalizeHist(image)
    addweighted_image = cv.addWeighted(image, config.alpha, equalized_image, config.beta, config.gamma)
    return addweighted_image

def crop_padding(image, radius, config):
    x_size = config.segmentation.padded_x_size
    y_size = config.segmentation.padded_y_size

    if x_size % 2 != 0 :
        x_size += 1
    if y_size % 2 != 0 :
        y_size += 1

    x_size /= 2
    y_size /= 2
    x = int(x_size - radius - 1)
    y = int(y_size - radius - 1)
    image = image[y: y + (2 * radius), x: x + (2 * radius)]
    centre = (x_size - x, y_size - y)
    resized_image, new_centre = padding(image, centre, config.segmentation.final_x_size, config.segmentation.final_y_size) 
    return resized_image, new_centre   