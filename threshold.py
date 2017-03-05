from methods import *
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

def thresh(img):
    res = apply_thresholds(img)

    binary = np.dstack((res*255, res*255, res*255))
    binary_warped = fast_unwarp_lane(binary)
    return binary_warped

    
if __name__ == "__main__":
    #inp = "project_small"
    inp = "project_video"

    process_video(infile=inp + ".mp4", 
              outfile=inp + "_threshold.mp4", 
              method=thresh)
