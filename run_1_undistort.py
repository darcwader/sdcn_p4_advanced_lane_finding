from methods import *
import cv2

#load calibration
mtx, dist = load_calibration()

test_image = cv2.imread('camera_cal/calibration1.jpg')
undist = cv2.undistort(test_image, mtx, dist)

plot_images_save('1_undistort.png', [ [test_image, undist] ])

