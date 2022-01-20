################
# this code takes snapshat video and extract face'images from these video
# This used to make dataset useing saudi famous people
################

#%%
import cv2

count_frames = 0
# -----Read Video-----------
video_path = "/Volumes/Lexar/DataScience/employee-attendance-facial-recognition/experiment_file/img/SS3.mp4"
cap= cv2.VideoCapture(video_path)
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while(cap.isOpened()):
    count_frames +=1
    print(count_frames)
    ret, frame = cap.read()
    faces_detected = face_haar_cascade.detectMultiScale(frame, 1.32, 5)
    for (x, y, w, h) in faces_detected:
        frame2 = frame[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        # -----Save frame as image-----------
        resize_img = cv2.resize(frame2, (250, 250))
        cv2.imwrite('img/Fasal_faces_resize_img/fasal3_' + str(count_frames) + '.jpg', resize_img)

cap.release()
cv2.destroyAllWindows()




