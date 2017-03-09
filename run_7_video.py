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
    undist = undistort(img)
    out = lane.sliding_window(img)
    out = fast_warp_lane(out)
    out = cv2.addWeighted(undist, 1.0, out, 0.5, 0)
    lane.draw_curvature(out)
    #lane.draw_search_window_area(out)
    lane.debug_image[0:720, 0:1280] = out
    cv2.rectangle(lane.debug_image,(0,0),(1280,1080),(0,255,255), 2)
    return lane.debug_image
    return out


if __name__ == "__main__":
    #inp = "project_small"
    #inp = "project_video"
    inp = "challenge_video"

    process_video(infile=inp + ".mp4",
              outfile=inp + "_final.mp4",
              method=process)
