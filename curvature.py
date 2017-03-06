from methods import *
from methods_sliding_window import *

l = Lane()

def frame(img):
    if l.left_fit == None:
        binary_warped = l.sliding_window(img)
        binary_filled = binary_warped
    else:
        binary_warped = l.sliding_window(img)
        binary_filled = l.projection(binary_warped)

    warp = fast_warp_lane(binary_filled)

    out = cv2.addWeighted(img, 1.0, warp, 0.5, 0)
    l.curvature(out)
    return out


if __name__ == "__main__":
    #inp = "project_small"
    inp = "project_video"

    process_video(infile=inp + ".mp4", 
            outfile=inp + "_curvature.mp4", 
            method=frame)
