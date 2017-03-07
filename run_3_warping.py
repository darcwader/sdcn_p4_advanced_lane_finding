from methods import *
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import glob
import cv2
import matplotlib.image as mpimg

def thresh(img, warp=False):
    res = apply_thresholds(img)

    binary = np.dstack((res*255, res*255, res*255))
    binary_warped = binary
    if warp == True:
        binary_warped = fast_unwarp_lane(binary)
    return binary_warped

    
if __name__ == "__main__":
    """
    #inp = "project_small"
    inp = "project_video"

    process_video(infile=inp + ".mp4", 
              outfile=inp + "_threshold.mp4", 
              method=thresh)
    """
    files = glob.glob("test_images/test2.jpg")

    print(files)
    images = []
    for ix, fname in enumerate(files):
        img = mpimg.imread(fname)
        
        t = thresh(img, False)
        images.append([img, t])

    plot_images_save("output_images/2_threshold.png", images)

    
