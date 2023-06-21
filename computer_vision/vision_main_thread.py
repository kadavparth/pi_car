#! usr/bin/env/python

import cv2 
import numpy as np 

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
            
            if ret and show:
                
                cv2.imshow('Raw Image',frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    
                    break

            else: 

                print("Check camera or set show = True")

if __name__ == "__main__":

    ll = Get_LaneLine()

    ll.show()
    
    cv2.destroyAllWindows()

    cap.release()