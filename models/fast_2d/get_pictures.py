###############
## This code used to take pictures form web came and then save it.
###############

import numpy as np
import cv2
import time
from datetime import datetime

frame_rate = 30
prev = 0
cap = cv2.VideoCapture(0)
count_frames = 0


while(True):
    cur_time = datetime.now()
    print("time:", cur_time)
    print("---------")
    time_elapsed = time.time() - prev
    if time_elapsed > 1. / frame_rate:
        count_frames += 1
        print(count_frames)
        #---------------------------------
        ret, frame = cap.read()
        prev = time.time()
        frame_h = 200*2
        frame_w = 300*2
        #-----Show Image-------------------
        resized_img = cv2.resize(frame, (frame_w, frame_h ))
        cv2.imshow('Facial Is Real', resized_img)

        #-----Save frame as image-----------
        cv2.imwrite('img/phone2_images/Real'+str(count_frames)+'.jpg',frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

###########################




