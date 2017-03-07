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
    out = lane.sliding_window(img)
    out = fast_warp_lane(out)
    out = cv2.addWeighted(img, 1.0, out, 0.5, 0)
    lane.draw_curvature(out)
    return out

    
if __name__ == "__main__":
    #inp = "project_small"
    inp = "project_video"

    process_video(infile=inp + ".mp4", 
              outfile=inp + "_px.mp4", 
              method=process)

    
