from methods import *

def frame(img):
    binary_warped = lane_boxes(img)
    return binary_warped
    
if __name__ == "__main__":
    #inp = "project_small"
    inp = "project_video"

    process_video(infile=inp + ".mp4", 
              outfile=inp + "_lane.mp4", 
              method=frame)
