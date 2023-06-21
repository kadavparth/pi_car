#! usr/bin/env/python

import cv2 
import numpy as np 

def threshold(channel, thresh=(128,255), thresh_type = cv2.THRESH_BINARY):

    return cv2.threshold(channel, thresh[0], thresh[1], thresh_type)

def blur_gaussian(channel, kernel_size=3):

    return cv2.GaussianBlur(channel, (kernel_size, kernel_size), 0)

def Perspective(img):

    """
    This is a function used to get the bird's eye view
    of the input image with given region of interest points
    """

    src = np.float32([[0,480], [180,200], [460,200], [640,480]])
    dst = np.float32([[0,img.shape[0]], [0,0], [img.shape[1], 0], [img.shape[1], img.shape[0]]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)

    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    unwarped = cv2.warpPerspective(img, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    return warped, Minv, unwarped


def histogram_peak(histogram):
    
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
 
    return leftx_base, rightx_base



def get_lane_line_indices_sliding_window(warped_frame,histogram):
    
    frame_sliding_window = warped_frame.copy()
    nwindows = 7
    margin = 25
    minpix = 50
    window_height = int(warped_frame.shape[0]/nwindows)
    
    nonzero = warped_frame.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1]) 
    
    left_lane_inds = []
    right_lane_inds = []
    
    leftx_base, rightx_base = histogram_peak(histogram)
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    for window in range(nwindows):
        
      win_y_low = warped_frame.shape[0] - (window + 1) * window_height
      win_y_high = warped_frame.shape[0] - window * window_height
      win_xleft_low = leftx_current - margin
      win_xleft_high = leftx_current + margin
      win_xright_low = rightx_current - margin
      win_xright_high = rightx_current + margin
      
      cv2.rectangle(frame_sliding_window,(win_xleft_low,win_y_low),(
        win_xleft_high,win_y_high), (255,255,255), 2)
      cv2.rectangle(frame_sliding_window,(win_xright_low,win_y_low),(
        win_xright_high,win_y_high), (255,255,255), 2)
 
      # Identify the nonzero pixels in x and y within the window
      good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (
                           nonzerox < win_xleft_high)).nonzero()[0]
      good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (
                            nonzerox < win_xright_high)).nonzero()[0]
                                                         
      # Append these indices to the lists
      left_lane_inds.append(good_left_inds)
      right_lane_inds.append(good_right_inds)
         
      # If you found > minpix pixels, recenter next window on mean position
      minpix = minpix
      
      if len(good_left_inds) > minpix:
        leftx_current = int(np.mean(nonzerox[good_left_inds]))
      if len(good_right_inds) > minpix:        
        rightx_current = int(np.mean(nonzerox[good_right_inds]))
        
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
 
    # Extract the pixel coordinates for the left and right lane lines
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds] 
    righty = nonzeroy[right_lane_inds]
 
    # Fit a second order polynomial curve to the pixel coordinates for
    # the left and right lane lines
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2) 
    
    ploty = np.linspace(0, frame_sliding_window.shape[0]-1, frame_sliding_window.shape[0])
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    midx = (left_fitx + right_fitx)/2
    
    out_img = np.dstack((frame_sliding_window, frame_sliding_window, 
                         (frame_sliding_window))) * 255

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 255, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    return left_fit, right_fit, left_fitx, right_fitx, midx, ploty ,out_img, frame_sliding_window
