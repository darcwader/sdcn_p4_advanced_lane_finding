import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from moviepy.editor import VideoFileClip
from tqdm import tqdm

def load_calibration():
    calib = pickle.load(open("calibration.pkl", "rb"))
    print("Calibration Loaded")
    return calib[0], calib[1]

#load calibration
mtx, dist = load_calibration()

img_size = (1280, 720)

def load_transforms():
    #load perspective vars
    src = np.float32([[  585.0-10,  460.0],
               [ 0.0,  720.0],
               [ 1280.0,  720.0],
               [ 695.0+10,  460.0]])

    dst = np.float32([[320, 0], 
                      [320, 720],
                      [960, 720], 
                        [960, 0]])
            
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

M, Minv = load_transforms()

def fast_unwarp_lane(img):
    global dist, mtx, img_size
    undist = cv2.undistort(img, mtx, dist)
    out = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
    return out

def fast_warp_lane(lane):
    global Minv, img_size
    unwarped = cv2.warpPerspective(lane, Minv, img_size, flags=cv2.INTER_LINEAR)
    return unwarped

def process_video(infile, outfile, method):
    """method has to accept rgb image and return rgb image. method is called on every frame of infile."""
    clip1 = VideoFileClip(infile)
    white_clip = clip1.fl_image(method) #NOTE: this function expects color images!!
    white_clip.write_videofile(outfile, audio=False)

def plot_images(images):
    """ Helper routine which plots all images passed as array in a single row """
    m = len(images)
    n = len(images[0])
    
    fig, axes = plt.subplots(m, n, figsize=(10*n, 10*m))
    fig.tight_layout()
    for ix in range(m):
        for iy in range(n):
            axes[ix][iy].imshow(images[ix][iy], cmap='gray')
            axes[ix][iy].axis('off')
    plt.show()


def abs_sobel_thresh(img_gray, orient='x', ksize=3,  thresh=(20,100)):    
    sobel = None
    if orient=='x':
        sobel = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=ksize)
    else:
        sobel = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_sobel = np.absolute(sobel)
    scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))

    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    
    return binary_output

def mag_thresh(gray, ksize=9, thresh=(20,80)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    abssolxy = np.sqrt(sobelx ** 2 + sobely ** 2)    
    scaledxy = (abssolxy*255/np.max(abssolxy)).astype(np.uint8)
    binary_output = np.zeros_like(scaledxy)
    binary_output[(scaledxy >= thresh[0]) & (scaledxy <= thresh[1])] = 1
    return binary_output
    
def dir_thresh(gray, ksize=15, thresh=(0.0, np.pi/2)):    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    abssobelx = np.absolute(sobelx)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    abssobely = np.absolute(sobely)
    
    abssobelxy = np.arctan2(abssobely,abssobelx)

    binary_output = np.zeros(abssobelxy.shape, dtype=np.uint8)
    binary_output[(abssobelxy >= thresh[0]) & (abssobelxy <= thresh[1])] = 1
    return binary_output

def hls_select(image_hsv):
    i_h = image_hsv[:,:,0]
    i_s = image_hsv[:,:,1]
    i_v = image_hsv[:,:,2]
    
    res = np.zeros_like(i_h).astype(np.uint8)
    res[((i_h > 0) & (i_h < 80)) & (i_s > 80) & (i_v > 100)]  = 1 #yellow only
    res[(i_s < 65) & (i_v > 200)]  = 1 #white only
    
    return res

def apply_stage_1(img_hsv):
    img_gray = img_hsv[:,:,1] #sobel operators don't work very well on rgb to isolate the lane lines. hsv they work very well
    
    x_image = abs_sobel_thresh(img_gray, orient='x', ksize=3, thresh=(20,200))
    y_image = abs_sobel_thresh(img_gray, orient='y', ksize=3, thresh=(20,200))
    xy_image = mag_thresh(img_gray, ksize=9, thresh=(20,100))
    dir_image = dir_thresh(img_gray, ksize=9, thresh=(0.7, 1.3))

    img_stage_1 = np.zeros_like(x_image)
    #img_stage_1[(x_image == 1) | ((xy_image == 1) & (dir_image == 1))]  = 1 #dir_image is not working after lot of trials and error.
    img_stage_1[((x_image == 1) | (y_image == 1))]  = 1
    return img_stage_1

def apply_thresholds(img):
    img = cv2.undistort(img, mtx, dist, None, mtx)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    stage_1 = apply_stage_1(img_hsv)
    stage_2 = hls_select(img_hsv)
    
    # hsv is generally better overall. x/y only on certain occasions.
    res = (stage_1 * 72) + (stage_2 * 182) # lot of trials, got this. hsv is prominent, but xy sobel is when hsv is not working.
    
    binary_out = np.zeros_like(res)
    binary_out[res > 100] = 1
    return binary_out

def lane_boxes(img):
    res = apply_thresholds(img)
    res_rgb = np.dstack((res*255, res*255, res*255))
    lane = fast_unwarp_lane(res_rgb)
    
    binary_warped = lane[:,:,2]

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    return out_img


