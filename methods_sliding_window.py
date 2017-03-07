
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from methods import *


class Line:
    def __init__(self):
        self.lane_inds = []
        # was the line detected in the last iteration?
        #self.detected = False
        # x values of the last n fits of the line
        #self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        #self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = np.array([False])
        self.previous_fit = []
        #radius of curvature of the line in some units
        self.radius_of_curvature = 0


        #distance in meters of vehicle center from the line
        #self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        #self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        #self.allx = None
        #y values for detected line pixels
        #self.ally = None

class Lane:
    
    def __init__(self):
        self.left = Line()
        self.right = Line()
    
    def binary_warped(self, img):
        res = apply_thresholds(img)
        res_rgb = np.dstack((res*255, res*255, res*255))
        lane = fast_unwarp_lane(res_rgb)
        
        binary_warped = lane[:,:,2]
        return binary_warped
    
    def sliding_window(self, img):
        binary_warped = self.binary_warped(img)
        
        if self.left.current_fit.size > 0:
            out = self.sliding_first(binary_warped)
        else:
            out = self.sliding_next(binary_warped)
        
        self.process_fits()
        out = self.draw_projection(binary_warped)

        return out


    def sliding_first(self, binary_warped):
        # Takins in binary warped, and returns sliding window drawn image with
        # left right inds colored
        
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
        self.left.lane_inds = []
        self.right.lane_inds = []

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
            self.left.lane_inds.append(good_left_inds)
            self.right.lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        self.left.lane_inds = np.concatenate(self.left.lane_inds)
        self.right.lane_inds = np.concatenate(self.right.lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[self.left.lane_inds]
        lefty = nonzeroy[self.left.lane_inds] 
        rightx = nonzerox[self.right.lane_inds]
        righty = nonzeroy[self.right.lane_inds] 

        # Fit a second order polynomial to each
        self.left.current_fit = np.polyfit(lefty, leftx, 2)
        self.right.current_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = self.left.current_fit[0]*ploty**2 + self.left.current_fit[1]*ploty + self.left.current_fit[2]
        right_fitx = self.right.current_fit[0]*ploty**2 + self.right.current_fit[1]*ploty + self.right.current_fit[2]

        out_img[nonzeroy[self.left.lane_inds], nonzerox[self.left.lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[self.right.lane_inds], nonzerox[self.right.lane_inds]] = [0, 0, 255]

        return out_img

    def sliding_next(self,binary_warped):
        # We now have a new warped binary image
        # It's now much easier to find line pixels!
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
    
        return binary_warped

    
    def draw_search_window_area(self, binary_warped):
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[self.left.lane_inds], nonzerox[self.left.lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[self.right.lane_inds], nonzerox[self.right.lane_inds]] = [0, 0, 255]
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = self.left.current_fit[0]*ploty**2 + self.left.current_fit[1]*ploty + self.left.current_fit[2]
        right_fitx = self.right.current_fit[0]*ploty**2 + self.right.current_fit[1]*ploty + self.right.current_fit[2]
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        return result

    def process_fits(self):
        last_n = 5
        
        self.left.previous_fit.append(self.left.current_fit)
        if len(self.left.previous_fit) > last_n:
            self.left.previous_fit = self.left.previous_fit[1:]
        self.l_best_fit = np.average(self.left.previous_fit, axis=0)
        
        self.right.previous_fit.append(self.right.current_fit)
        if len(self.right.previous_fit) > last_n:
            self.right.previous_fit = self.right.previous_fit[1:]
        self.r_best_fit = np.average(self.right.previous_fit, axis=0)

        self.left.current_fit = self.l_best_fit
        self.right.current_fit = self.r_best_fit

    
    def draw_curvature(self, img):
        #draws curvature metrics onto the img
        y_eval = np.max(img.shape[0]-1)
        left_curverad = ((1 + (2*self.left.current_fit[0]*y_eval + self.left.current_fit[1])**2)**1.5) / np.absolute(2*self.left.current_fit[0])
        right_curverad = ((1 + (2*self.right.current_fit[0]*y_eval + self.right.current_fit[1])**2)**1.5) / np.absolute(2*self.right.current_fit[0])
        #cv2.putText(img, "left:{0:.2f}".format(left_curverad), (100,100), cv2.FONT_HERSHEY_PLAIN,2, 255)
        #cv2.putText(img, "right:{0:.2f}".format(right_curverad), (100,150), cv2.FONT_HERSHEY_PLAIN,2, 255)
        #print(left_curverad, right_curverad)


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
        # Now our radius of curvature is in meters
        #print(left_curverad, 'm', right_curverad, 'm')
        
        cv2.putText(img, "Radius Left:{0:.2f}m".format(self.left.radius_of_curvature), (10,50), cv2.FONT_HERSHEY_PLAIN, 2, 255)
        cv2.putText(img, "Radius Right:{0:.2f}m".format(self.right.radius_of_curvature), (10,100), cv2.FONT_HERSHEY_PLAIN, 2, 255)
        # Example values: 632.1 m    626.2 m
        return img


    def draw_projection(self, binary_warped):
        #draws the projection and returns color image
        
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = self.left.current_fit[0]*ploty**2 + self.left.current_fit[1]*ploty + self.left.current_fit[2]
        right_fitx = self.right.current_fit[0]*ploty**2 + self.right.current_fit[1]*ploty + self.right.current_fit[2]

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        return color_warp
        
