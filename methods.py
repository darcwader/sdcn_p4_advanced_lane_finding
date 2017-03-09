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

def undistort(img):
    undist = cv2.undistort(img, mtx, dist)
    return undist

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

def plot_images_save(fname, images):
    """ Helper routine which plots all images passed as array in a single row """
    print(len(images))
    m = len(images)
    n = len(images[0])
    
    fig, axes = plt.subplots(m, n, figsize=(10*n, 10*m))
    if m == 1:
        axes = [axes]
    fig.tight_layout()
    for ix in range(m):
        for iy in range(n):
            axes[ix][iy].imshow(images[ix][iy], cmap='gray')
            axes[ix][iy].axis('off')
    fig.savefig(fname)

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

def hsv_select(image_hsv):
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
    stage_2 = hsv_select(img_hsv)
    
    # hsv is generally better overall. x/y only on certain occasions.
    res = (stage_1 * 72) + (stage_2 * 182) # lot of trials, got this. hsv is prominent, but xy sobel is when hsv is not working.
    
    binary_out = np.zeros_like(res)
    binary_out[res > 100] = 1
    return binary_out

