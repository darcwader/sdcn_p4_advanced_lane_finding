
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/1_undistorted.png "Undistorted"
[image2]: ./output_images/2_threshold.png "Theshold"
[image3]: ./output_images/3_warping.png "Warping"
[image4]: ./output_images/4_process.png "Sliding Window"
[image6]: ./output_images/6_lane.png "Projection"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this is in `calibrate.py`. this saves the calibration to a pickle file.
method to load is in `methods.py` along with other helper routines. 

First all points are created using `np.mgrid` and `cv2.findChessboardCorners` is called on each image in `camera_cal/*.jpg` files.

All points are then collected into `objpoints` and `imgpoints`. the images are also plotted to show make sure it finds everything properly.


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To test if this is fine run `run_1_undistort.py` is run which saves file into `output_images/1_undistor.py`

![Undistorted][image1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

*Transform*

`methods.py` contains function `load_transforms` which loads transform matrix. `fast_unwarp_lane` and `fast_warp_lane` then use the loaded matrix to perform the warp and unwarping.


*Thresholding* 

This was the hardest part of the entire project, took weeks.  

Unsuccessful tries: 

   * I tried sobel x filter , sobel y filter, xy filter, dir filter on RGB image. Although it went well on image, further down the pipeline this failed badly.
   * I used HSV color space and it produced nice results along with sobelx, xy, dir in RGB space. but realized it also failed in some cases.

Successful

   * I took some snapshots of track and applied filter on then instead of test images.
   * I converted to HSV only, RGB was just not helpful
   * on HSV filter out higher thresholds to isolate lane lines
   * on HSV filter also apply weights to H, S, V to get some addional information
   * threshold the above two to get a binary lane

Code is in `methods.py`


I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

method that applies the threshold is `apply_thresholds` and returns a unwarped binary file.

![alt text][image2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The transform is performed by using the folowing

```
src = np.float32([[  585.0-10,  460.0],
                [ 0.0,  720.0],
                [ 1280.0,  720.0],
                [ 695.0+10,  460.0]])

dst = np.float32([[320, 0],
                [320, 720],
                [960, 720],
                [960, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575.0, 460    | 320, 0        | 
| 0, 720        | 320, 720      |
| 1280, 720     | 960, 720      |
| 705.0, 460    | 960, 0        |

I verified that my perspective transform was working as expected. here is the warped lane drawing.

![alt text][image3]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

the code for lane and all details are in `methods_sliding_window.py`

we have a `Lane` class. 

In this we find the lane lines by using histogram of thresolded image. We find the base by finding the max 

```
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
```

Then we divide into small strips and calculate the highest values in within a certain margin.

resulting windows if plotted would show.

![alt text][image4]


This is only done for the first frame.

Successive frames we use `sliding_next` methods in `Lane` class. Which uses the previous fits to calculate the next frames.

```
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
margin = 100
self.left.lane_inds = ((nonzerox > (self.left.current_fit[0]*(nonzeroy**2) + self.left.current_fit[1]*nonzeroy + self.left.current_fit[2] - margin)) & (nonzerox < (self.left.current_fit[0]*(nonzeroy**2) + self.left.current_fit[1]*nonzeroy + self.left.current_fit[2] + margin)))
self.right.lane_inds = ((nonzerox > (self.right.current_fit[0]*(nonzeroy**2) + self.right.current_fit[1]*nonzeroy + self.right.current_fit[2] - margin)) & (nonzerox < (self.right.current_fit[0]*(nonzeroy**2) + self.right.current_fit[1]*nonzeroy + self.right.current_fit[2] + margin)))  

# Again, extract left and right line pixel positions
leftx = nonzerox[self.left.lane_inds]
lefty = nonzeroy[self.left.lane_inds] 
rightx = nonzerox[self.right.lane_inds]
righty = nonzeroy[self.right.lane_inds]
# Fit a second order polynomial to each
self.left.current_fit = np.polyfit(lefty, leftx, 2)
self.right.current_fit = np.polyfit(righty, rightx, 2)
```

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

radius of curvature is calculated using `draw_curvature` methods in `Lane` class.

```
y_eval = np.max(img.shape[0]-1)
left_curverad = ((1 + (2*self.left.current_fit[0]*y_eval + self.left.current_fit[1])**2)**1.5) / np.absolute(2*self.left.current_fit[0])
right_curverad = ((1 + (2*self.right.current_fit[0]*y_eval + self.right.current_fit[1])**2)**1.5) / np.absolute(2*self.right.current_fit[0])

ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
leftx = self.left.current_fit[0]*ploty**2 + self.left.current_fit[1]*ploty + self.left.current_fit[2]
rightx = self.right.current_fit[0]*ploty**2 + self.right.current_fit[1]*ploty + self.right.current_fit[2]

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Fit new polynomials to x,y in world space
left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

# Calculate the new radii of curvature
self.left.radius_of_curvature = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
self.right.radius_of_curvature  = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is done in `draw_projection` in `Lane` class from `methods_sliding_window.py`

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).


The lane line is smoothed over 5 best fit's to remove jitters.

Here's a [link to my video result](./output_images/project_video_final.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* the lane lines histogram can fail if the curvature is too high. 
* brightness vairations are not recorded well
* the the surface of road is more brighter the thresholding can fail and not find the lanes
* average of lane lines might not be good enough fit. 
* the left and right lane movement is not related in my algorithm. convolution was a better method, maybe that should be adopted for better performance.



