#! usr/bin/env/python

import cv2 
import numpy as np 
import utils as ut

class Get_LaneLine:

    def __init__(self):
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
            
            _, binary = ut.threshold(frame1[:,:,1], thresh=(180,240))

            warped, Minv, unwarped = ut.Perspective(binary)

            if show:
                
                cv2.imshow('Raw Image',frame)                
                
                cv2.imshow('Warped',warped)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    
                    break

            else: 

                print("Check camera or set show = True")

if __name__ == "__main__":

    ll = Get_LaneLine()

    ll.show()
    
    cv2.destroyAllWindows()

    cap.release()