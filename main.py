import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from tqdm import tqdm

def load_calibration():
    calib = pickle.load(open("calibration.pkl", "rb"))
    print("Calibration Loaded")
    return calib[0], calib[1]

mtx, dist = load_calibration()

#load calibration
img_size = (1280, 720)
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

def fast_unwarp_lane(img):
    undist = cv2.undistort(img, mtx, dist)
    out = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
    return out

def fast_warp_lane(lane):
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
    img_gray = img_hsv[:,:,1]
    
    x_image = abs_sobel_thresh(img_gray, orient='x', ksize=3, thresh=(20,200))
    y_image = abs_sobel_thresh(img_gray, orient='y', ksize=3, thresh=(20,200))
    xy_image = mag_thresh(img_gray, ksize=9, thresh=(20,100))
    dir_image = dir_thresh(img_gray, ksize=9, thresh=(0.7, 1.3))

    img_stage_1 = np.zeros_like(x_image)
    #img_stage_1[(x_image == 1) | ((xy_image == 1) & (dir_image == 1))]  = 1
    img_stage_1[((x_image == 1) | (y_image == 1))]  = 1
    return img_stage_1

def apply_thresholds(img):
    img = cv2.undistort(img, mtx, dist, None, mtx)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    stage_1 = apply_stage_1(img_hsv)
    stage_2 = hls_select(img_hsv)
    
    return stage_1, stage_2
    """
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    masked_image = cv2.bitwise_and(image,image, mask=mask)
    return mask, masked_image
    #return white_mask, cv2.bitwise_and(image,image, mask=white_mask) to investigate only on
    """

#plot the thresholded images
"""
imagenames = glob.glob('snapshots/*.png')

lane_images = []

for fname in tqdm(imagenames):
    img = mpimg.imread(fname)

    res1, res2 = apply_thresholds(img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    res1_rgb = np.dstack((res1*255, res1*255, res1*255))
    res2_rgb = np.dstack((res2*255, res2*255, res2*255))

    res1_uw = fast_unwarp_lane(res1_rgb)
    res2_uw = fast_unwarp_lane(res2_rgb)

    
    #plot_images([img, res1, out])
    lane_images.append([fast_unwarp_lane(img_hsv[:,:,1]), fast_unwarp_lane(img_hsv[:,:,2]), res1_uw, res2_uw])

plot_images(lane_images)
"""

# plotting is done, try on video.
#inp = "project_video"
#inp = "project_small"
#inp = "challenge_video"
#inp = "harder_challenge_video"


def lane_boxes(img):
    res1, res2 = apply_thresholds(img)
    res = (res1 * 72) + (res2 * 182)

    res2_rgb = np.dstack((res, res, res))
    res2_uw = fast_unwarp_lane(res2_rgb)

    return res2_uw

process_video(infile=inp + ".mp4", 
              outfile=inp + "_px_2.mp4", 
              method=lane_boxes)

# inference. looks like both red, green together do a good job of identifying lanes.
# hsv is generally better overall. x/y only on certain occasions.




