#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, InputLayer
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
import tensorflow as Tensor_Bord
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor='val_accuracy', min_delta= 0.01 , patience= 25, verbose= 1, mode='auto')
mc = ModelCheckpoint(filepath="wrapped_CNN_model4.h5", monitor= 'val_accuracy', verbose= 1, save_best_only= True, mode = 'auto')
nn_TensorBoard = Tensor_Bord.keras.callbacks.TensorBoard(log_dir="logs/wrapped_CNN_model4", histogram_freq=1)

#%%
files_path_list = [
    '/Volumes/Lexar/DataScience/employee-attendance-facial-recognition/experiment_file/img/my_Wrapped_images',
    '/Volumes/Lexar/DataScience/employee-attendance-facial-recognition/experiment_file/img/phone_Wrapped_images',
]


#%%
img_data_array = []
class_name = []
for index, file_path in enumerate(files_path_list):
    for img in os.listdir(file_path):  #Bring all images in this folder
        if img.endswith('.jpg') and '._' not in img:
            path = f"{file_path}/{img}"
            image = np.array(Image.open(path))
            image = np.resize(image, (144, 256, 3))
            image = image.astype('float32')
            image /= 255
            img_data_array.append(image)
            class_name.append(index)

#%%
# print(img_data_array)
img_arr = np.array(img_data_array, np.float32)
print(img_arr.shape)
class_arr = np.array(class_name, np.float32)
print(class_arr.shape)
class_arr_cat = to_categorical(class_arr)
# print(class_arr_cat)

###################
###     CNN    ####
###################

#%%
X_train, X_test, y_train, y_test = (train_test_split(img_arr, class_arr_cat, test_size = .15, random_state = 42))

#%%
NN = Sequential()
NN.add(InputLayer(input_shape=X_train.shape[1:]))
NN.add(Conv2D(filters=80, kernel_size=3, activation='relu', padding='same'))
NN.add(MaxPooling2D())
NN.add(Conv2D(filters=90, kernel_size=3, activation='relu', padding='same'))
NN.add(MaxPooling2D())
NN.add(Conv2D(filters=110, kernel_size=3, activation='relu', padding='same'))
NN.add(GlobalAveragePooling2D())
NN.add(Dense(240, activation='relu'))
NN.add(Dropout(0.4))
NN.add(Dense(200, activation='relu'))
NN.add(Dropout(0.3))
NN.add(Dense(170, activation='relu'))
NN.add(Dropout(0.3))
NN.add(Dense(len(files_path_list), activation='softmax'))  # Target classes will be same as len of paths
#%%
model = Sequential()
inputShape = (144, 256, 3)
chanDim = -1

# if we are using "channels first", update the input shape
# and channels dimension
# if K.image_data_format() == "channels_first":
#     inputShape = (depth, height, width)
#     chanDim = 1

#%%
from tensorflow.keras.layers import Activation
NN = Sequential()
NN.add(InputLayer(input_shape=inputShape))
NN.add(Conv2D(filters=80, kernel_size=3, activation='relu', padding='same'))
NN.add(MaxPooling2D())
NN.add(Conv2D(filters=90, kernel_size=3, activation='relu', padding='same'))
NN.add(MaxPooling2D())
NN.add(Conv2D(filters=110, kernel_size=3, activation='relu', padding='same'))
NN.add(GlobalAveragePooling2D())
NN.add(Dense(240, activation='relu'))
NN.add(Dropout(0.4))
NN.add(Dense(200, activation='relu'))
NN.add(Dropout(0.3))
NN.add(Dense(170, activation='relu'))
NN.add(Dropout(0.3))
NN.add(Dense(len(files_path_list)))
NN.add(Activation("softmax"))

#%%
print(NN.summary())
from tensorflow.keras.utils import plot_model
plot_model(NN, "model.png", show_shapes=True)

#%%
NN.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'],
)
NN.fit(X_train, y_train, epochs=300, verbose=1,  callbacks=[nn_TensorBoard ,es , mc ],  validation_split=0.20,)  #Track progress as we fit

#%%
NN.evaluate(X_test , y_test)
# #%%
# y_pred = NN.predict(X_test)

##############
##SAVE MODEL##
#%%
from keras.models import load_model
# NN.save('wrapped_CNN_model.h5')  # creates a HDF5 file
# del model  # deletes the existing model

# returns a compiled model
# model = load_model('my_model.h5')

#%%
# Epoch 100/100
# 157/157 [==============================] - 36s 230ms/step - loss: 0.1403 - accuracy: 0.9470 - val_loss: 0.1173 - val_accuracy: 0.9554
# NN.evaluate(X_test , y_test)
# 35/35 [==============================] - 2s 47ms/step - loss: 0.0996 - accuracy: 0.9684
# [0.09957810491323471, 0.9683830142021179]


# Epoch 21/300
# 157/157 [==============================] - ETA: 0s - loss: 0.0757 - accuracy: 0.9737
# Epoch 00021: val_accuracy improved from 0.96972 to 0.97371, saving model to wrapped_CNN_model2.h5
# 157/157 [==============================] - 105s 669ms/step - loss: 0.0757 - accuracy: 0.9737 - val_loss: 0.0715 - val_accuracy: 0.9737
# Epoch 00021: early stopping


# Epoch 67/300
# 157/157 [==============================] - ETA: 0s - loss: 6.9119e-04 - accuracy: 1.0000
# Epoch 00067: val_accuracy improved from 0.98247 to 0.98327, saving model to wrapped_CNN_model3.h5
# 157/157 [==============================] - 168s 1s/step - loss: 6.9119e-04 - accuracy: 1.0000 - val_loss: 0.0846 - val_accuracy: 0.9833





# Epoch 00048: val_accuracy improved from 0.97689 to 0.97928, saving model to wrapped_CNN_model4.h5
# 157/157 [==============================] - 217s 1s/step - loss: 0.0136 - accuracy: 0.9952 - val_loss: 0.0935 - val_accuracy: 0.9793
# Epoch 00057: early stopping
# 35/35 [==============================] - 11s 301ms/step - loss: 0.0674 - accuracy: 0.9828



# #%%
# NN = Sequential()
# NN.add(InputLayer(input_shape=X_train.shape[1:]))
# NN.add(Conv2D(filters=80, kernel_size=3, activation='relu', padding='same'))
# NN.add(MaxPooling2D())
# NN.add(Conv2D(filters=100, kernel_size=3, activation='relu', padding='same'))
# NN.add(MaxPooling2D())
# NN.add(Conv2D(filters=120, kernel_size=3, activation='relu', padding='same'))
# NN.add(GlobalAveragePooling2D())
# NN.add(Dense(450, activation='relu'))
# NN.add(Dropout(0.4))
# NN.add(Dense(400, activation='relu'))
# NN.add(Dropout(0.3))
# NN.add(Dense(350, activation='relu'))
# NN.add(Dropout(0.3))
# NN.add(Dense(len(files_path_list), activation='softmax'))  # Target classes will be same as len of paths
# #%%
# NN.compile(
#     loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'],
# )
# nn_TensorBoard = Tensor_Bord.keras.callbacks.TensorBoard(log_dir="logs/FFT_CNN", histogram_freq=1)
# NN.summary()
# NN.fit(X_train, y_train, epochs=350, verbose=1,  callbacks=[nn_TensorBoard],  validation_split=0.20,)  #Track progress as we fit
# #%%
# NN.evaluate(X_test , y_test)
# #%%
# y_pred = NN.predict(X_test)
#






