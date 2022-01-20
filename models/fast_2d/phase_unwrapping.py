###################
# This code will take all image and wraaped them to new type of images
###################

#%%
import numpy as np
from matplotlib import pyplot as plt
from skimage import data, img_as_float, color, exposure
from skimage.restoration import unwrap_phase
from PIL import Image
import cv2
import os

#%%
file_path = '/Volumes/Lexar/DataScience/employee-attendance-facial-recognition/experiment_file/img/phone2_images'
for img in os.listdir(file_path):
    print(f"{file_path}/{img}")
    img1 = Image.open(f"{file_path}/{img}")
    image = color.rgb2gray(img_as_float(img1))
    # Scale the image to [0, 4*pi]
    image = exposure.rescale_intensity(image, out_range=(0, 4 * np.pi))
    # Create a phase-wrapped image in the interval [-pi, pi)
    image_wrapped = np.angle(np.exp(1j * image))
    visual = np.array(image_wrapped)
    visual = (visual - visual.min()) / (visual.max() - visual.min())
    result = Image.fromarray((visual * 255).astype(np.uint8))
    result.save(f'img/phone2_Wrapped_images/wrapped_phone2_{img}')


