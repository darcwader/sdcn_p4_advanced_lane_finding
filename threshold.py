from main import *
import glob
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from tqdm import tqdm

if __name__ == "__main__":
    files = glob.glob("snapshots/*.png")

    images = []
    for fn in files:
        img = mpimg.imread(fn)

        out = apply_thresholds(img)
        rgb = np.dstack((out*255, out*255, out*255))
        uw = fast_unwarp_lane(rgb)
        images.append([img, out, uw])

    plot_images(images)

