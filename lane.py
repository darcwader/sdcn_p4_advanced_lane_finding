from methods import *
from slidingwindow import *

l = Lane()

def frame(img):
    binary_warped = l.sliding_window(img)

    warp = fast_warp_lane(binary_warped)

    out = cv2.addWeighted(img, 1.0, warp, 0.5, 0)
    return out

def frame_convolution(img):
    res = apply_thresholds(img)
    res_rgb = np.dstack((res*255, res*255, res*255))
    lane = fast_unwarp_lane(res_rgb)
    unwarp = convolution(lane[:,:,0])
    warp = fast_warp_lane(unwarp)	

    out = cv2.addWeighted(img, 1.0, warp, 0.5, 0)

if __name__ == "__main__":
    inp = "project_small"
    #inp = "project_video"

    process_video(infile=inp + ".mp4", 
            outfile=inp + "_final.mp4", 
            method=frame_convolution)
