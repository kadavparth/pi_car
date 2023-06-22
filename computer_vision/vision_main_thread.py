#! usr/bin/env/python

import cv2 
import numpy as np 
import utils as ut
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from matplotlib import style

class Get_LaneLine:

    def __init__(self):

        try:
            self.show()

        except:
            pass

    def show(self, CAM_OBJECT : int = 0, show = True):

        """
        Output the raw camera feed
        show = True by default
        
        """
    
        global cap 
        
        cap = cv2.VideoCapture(CAM_OBJECT)

        while True:
            
            ret, frame = cap.read()

            frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
            
            _, binary = ut.threshold(frame1[:,:,2], thresh=(200,240))

            warped, Minv, unwarped = ut.Perspective(binary)

            histogram = ut.calc_hist(warped)

            left_fit, right_fit, left_fitx, right_fitx, midx, ploty ,out_img, frame_sliding_window = ut.get_lane_line_indices_sliding_window(warped, histogram)

            blank_img = np.zeros_like(frame)

            if show:
                
                cv2.imshow('Raw Image',frame1)                
                
                cv2.imshow('Warped',warped)

                for i in range(len(ploty)):

                    cv2.line(blank_img, (int(left_fitx[i]), int(ploty[i])), (int(left_fitx[i]), int(ploty[i])), (0,255,0),3)
                    cv2.line(blank_img, (int(right_fitx[i]), int(ploty[i])), (int(right_fitx[i]), int(ploty[i])), (0,0,255),3)
                    cv2.line(blank_img, (int(midx[i]), int(ploty[i])), (int(midx[i]), int(ploty[i])), (255,255,255),3)

                cv2.imshow('Sliding Window', blank_img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    
                    break

            else: 

                print("Check camera or set show = True")


if __name__ == "__main__":

    ll = Get_LaneLine()
    
    cv2.destroyAllWindows()

    cap.release()