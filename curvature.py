from methods import *
from methods_sliding_window import *

l = Lane()

def frame(img):
    binary_warped = l.sliding_window(img)
    warp = fast_warp_lane(binary_warped)

    out = cv2.addWeighted(img, 1.0, warp, 0.5, 0)
    l.draw_curvature(out)
    return out


if __name__ == "__main__":
    inp = "project_small"
    #inp = "project_video"

    process_video(infile=inp + ".mp4", 
            outfile=inp + "_curvature.mp4", 
            method=frame)
