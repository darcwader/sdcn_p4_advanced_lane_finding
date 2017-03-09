from methods import *
import cv2

#load calibration
mtx, dist = load_calibration()

test_image = cv2.imread('test_images/test4.jpg')
test_image = test_image[..., ::-1]
undist = cv2.undistort(test_image, mtx, dist)

plot_images_save('output_images/1b_undistort.png', [ [test_image, undist] ])

