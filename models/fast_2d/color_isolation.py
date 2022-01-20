import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow, imread
from skimage.color import rgb2hsv, hsv2rgb
import cv2


#################
# trying to extract basic colors ( 'Reds','Greens','Blues' ) for each image
#################
#%%
for x in range(10):
    print(x)
    path = f'/Volumes/Lexar/DataScience/employee-attendance-facial-recognition/experiment_file/img/my_images/Real1{x}0.jpg'
    # path = f'/Volumes/Lexar/DataScience/employee-attendance-facial-recognition/experiment_file/img/phone_images/Real14{x}9.jpg'
    red_girl = imread(path)
    def rgb_splitter(image):
        rgb_list = ['Reds','Greens','Blues']
        fig, ax = plt.subplots(1, 3, figsize=(15,5), sharey = True)
        for i in range(3):
            ax[i].imshow(image[:,:,i], cmap = rgb_list[i])
            ax[i].set_title(rgb_list[i], fontsize = 15)
    rgb_splitter(red_girl)

plt.show()

########################
##Convert image  to three new images with different color: 'Hue', 'Saturation', 'Value'
########################

#%%
def display_as_hsv(image):
    img = cv2.imread(image)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv_list = ['Hue', 'Saturation', 'Value']
    fig, ax = plt.subplots(1, 3, figsize=(15, 7), sharey=True)

    ax[0].imshow(img_hsv[:, :, 0], cmap='hsv')
    ax[0].set_title(hsv_list[0], fontsize=20)
    ax[0].axis('off')

    ax[1].imshow(img_hsv[:, :, 1], cmap='Greys')
    ax[1].set_title(hsv_list[1], fontsize=20)
    ax[1].axis('off')

    ax[2].imshow(img_hsv[:, :, 2], cmap='gray')
    ax[2].set_title(hsv_list[2], fontsize=20)
    ax[2].axis('off')

    fig.tight_layout()

#%%
path = '/Volumes/Lexar/DataScience/employee-attendance-facial-recognition/experiment_file/img/my_images/Real721.jpg'
# path = '/Volumes/Lexar/DataScience/employee-attendance-facial-recognition/experiment_file/img/phone_images/Real829.jpg'
display_as_hsv(path)
plt.show()