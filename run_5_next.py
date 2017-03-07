from methods import *
from methods_sliding_window import *
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import glob
import cv2
import matplotlib.image as mpimg

lane = Lane()

def process(img):
    out = lane.binary_warped(img)
    out = lane.sliding_first(out)
    lane.process_fits()
    out = lane.sliding_next(out)
    out = lane.draw_search_window_area(out)
    return out

    
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
        
        t = process(img)
        images.append([img, t])

    plot_images_save("output_images/5_sliding.png", images)

    
