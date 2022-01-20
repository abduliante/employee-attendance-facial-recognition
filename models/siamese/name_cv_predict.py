######################
# This is our final porduct, we merged fack/real Model with name predcation modal
######################

#%%
import numpy as np
import cv2
from keras.models import  load_model
from PIL import Image
from skimage import data, img_as_float, color, exposure
import tensorflow as tf

#%%
cap = cv2.VideoCapture(0)
count_frames = 0
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model_path = "/Volumes/Lexar/DataScience/employee-attendance-facial-recognition/experiment_file/siamese_model3.h5"
model = load_model(model_path, compile=False )


model_path1 = "/Volumes/Lexar/DataScience/employee-attendance-facial-recognition/experiment_file/wrapped_CNN_model2.h5"
model1 = load_model(model_path1)

#%%
img_path_list = [
    "/Volumes/Lexar/our_images/mohammed.jpeg" ,
    "/Volumes/Lexar/our_images/hatim.jpeg" ,
    "/Volumes/Lexar/our_images/abdulrahman.jpeg"
]
names_list = ['Mohammed' , 'Hatim' ,  'Abdulrahman' ]
# names_list = ['Mohammed' , 'Hatim' ,   'Mohammed']

def scale_images(one_img_path):
    image2 = np.array(Image.open(one_img_path))
    image2 = np.resize(image2, (1 ,100, 100, 3))
    image2 = image2.astype('float32')
    image2 /= 255
    return image2

img_list_scled = []
for img_path in img_path_list :
    img_list_scled.append(scale_images(img_path))



#%%
while(True):
    count_frames += 1
    print(count_frames)
    #---------------------------------
    ret, frame = cap.read()
    frame_h = 500
    frame_w = 800

    #-----Drawing Rectangle------------
    start_point = (55, 55)# represents the top left corner of rectangle
    end_point = (frame_w + 420, frame_h + 200 )# represents the bottom right corner of rectangle
    thickness = 9 # Line thickness of 2 px


    ################################
    ####   Predict Real Fack    ####
    ################################
    image = color.rgb2gray(img_as_float(frame))
    image = exposure.rescale_intensity(image, out_range=(0, 4 * np.pi))
    image_wrapped = np.angle(np.exp(1j * image))
    visual = np.array(image_wrapped)
    visual = (visual - visual.min()) / (visual.max() - visual.min())
    result = Image.fromarray((visual * 255).astype(np.uint8))
    # visual2 = np.resize(result, (1,  120, 213, 3))
    visual2 = np.resize(result, (1,  120, 213, 3))
    visual2 = visual2.astype('float32')
    visual2 /= 255
    y_pred = model1.predict(visual2)
    emotions = ['Real', 'Fake']
    predicted_emotion = emotions[np.argmax(y_pred)]
    print(predicted_emotion)
    if predicted_emotion == 'Fake' :
        cv2.rectangle(frame, start_point, end_point, (0, 0 , 255), thickness)
    if predicted_emotion == "Real":
        cv2.rectangle(frame, start_point, end_point, (0, 255 , 0), thickness)



    #-----Write Text-------------------
    font = cv2.FONT_HERSHEY_SIMPLEX #denotes the font type
    org = (60, 130) #coordinates of the bottom-left corner of the text string in the image.
    fontScale = 2# fontScale
    color_text = (255, 0, 0)
    thickness = 3 # Line thickness
    text = str(predicted_emotion)
    cv2.putText(frame, text, org, font, fontScale, color_text, thickness, cv2.LINE_AA)


    ################################
    ####  Predict Name ##
    ################################
    faces_detected = face_haar_cascade.detectMultiScale(frame, 1.32, 5)
    for (x, y, w, h) in faces_detected:
        frame2 = frame[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        #-----Predict from Model----------
        image1 = np.array(frame2)
        image1 = np.resize(image1, (1, 100, 100, 3))
        image1 = image1.astype('float32')
        image1 /= 255
        tem_result = []
        tem_name = []
        for idx , img in enumerate(img_list_scled):
            prediction = model.predict([image1, img_list_scled[idx]])
            print(f"{prediction[0][0]} , {names_list[idx]} ")
            tem_name.append(names_list[idx])
            tem_result.append(prediction[0][0])
        tem_result = np.array(tem_result)
        large_index = np.argmax(tem_result)

        #-----Write Text-------------------
        font = cv2.FONT_HERSHEY_SIMPLEX #denotes the font type
        org = (60, 130) #coordinates of the bottom-left corner of the text string in the image.
        fontScale = 2# fontScale
        color_text = (255, 0, 0)
        text = str(tem_name[large_index])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (0, 255, 0), -2)
        cv2.putText(frame, text, (x, y - 10), font, 0.75, (255, 0, 0), 1, cv2.LINE_AA)


    #-----Show Image-------------------
    resized_img = cv2.resize(frame, (frame_w, frame_h ))
    cv2.imshow('Facial Is Real', resized_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

