#%%
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from PIL import Image
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as Tensor_Bord
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


es = EarlyStopping(monitor='val_accuracy', min_delta= 0.01 , patience= 8, verbose= 1, mode='auto')
mc = ModelCheckpoint(filepath="siamese_model4.h5", monitor= 'val_accuracy', verbose= 1, save_best_only= True, mode = 'auto')
nn_TensorBoard = Tensor_Bord.keras.callbacks.TensorBoard(log_dir="logs/siamese_model4", histogram_freq=1)


#%%
file_path = '/Volumes/Lexar/Downloads/105_classes_pins_dataset'
# file_path = '/Volumes/Lexar/DataScience/employee-attendance-facial-recognition/experiment_file/img/Faces'
img_data_array = []
class_name = []
for index,  file_name in enumerate(os.listdir(file_path)):
    for img in os.listdir(f'{file_path}/{file_name}'):
        if img.endswith('.jpg') and '._' not in img:
            path = f"{file_path}/{file_name}/{img}"
            image = np.array(Image.open(path))
            image = np.resize(image, (100, 100, 3))
            image = image.astype('float32')
            image /= 255
            img_data_array.append(image)
            class_name.append(int(index))

#%%
img_data_array = np.array(img_data_array, np.float32)
class_name = np.array(class_name)
#%%
x_train, x_test, y_train, y_test = (train_test_split(img_data_array, class_name, test_size = .15, random_state = 42))

#%%
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

#%%
x_train, x_val, y_train, y_val = (train_test_split(x_train, y_train, test_size = .25, random_state = 42))
#%%
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
print(y_val[0:9])

########################
# Create pairs of images
########################
#%%
def make_pairs(x, y):
    num_classes = int(max(y)) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]
    print(digit_indices)
    pairs = []
    labels = []
    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]
        pairs += [[x1, x2]]
        labels += [1]
        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

    return np.array(pairs), np.array(labels).astype("float32")


#%%
pairs_train, labels_train = make_pairs(x_train, y_train)
pairs_val, labels_val = make_pairs(x_val, y_val)
pairs_test, labels_test = make_pairs(x_test, y_test)

#%%
x_train_1 = pairs_train[:, 0]
x_train_2 = pairs_train[:, 1]
x_val_1 = pairs_val[:, 0]
x_val_2 = pairs_val[:, 1]
x_test_1 = pairs_test[:, 0]
x_test_2 = pairs_test[:, 1]
#%%

def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


input = layers.Input((100, 100, 3))
x = tf.keras.layers.BatchNormalization()(input)
x = layers.Conv2D(filters=50, kernel_size=3, activation='relu', padding='same')(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(filters=80, kernel_size=3, activation='relu', padding='same')(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(filters=90, kernel_size=3, activation='relu', padding='same')(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = layers.Dense(150, activation='relu')(x)
x = layers.Dense(190, activation='relu')(x)
x = layers.Dense(120, activation='relu')(x)

embedding_network = keras.Model(input, x)


input_1 = layers.Input((100, 100, 3))
input_2 = layers.Input((100, 100, 3))

# As mentioned above, Siamese Network share weights between
# tower networks (sister networks). To allow this, we will use
# same embedding network for both tower networks.
tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)

#%%
def loss(margin=1):
    def contrastive_loss(y_true, y_pred):
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )
    return contrastive_loss

#%%
epochs = 50
batch_size = 16
margin = 1  # Margin for constrastive loss.
siamese.compile(loss=loss(margin=margin), optimizer="adam", metrics=["accuracy"])
siamese.summary()

#%%
history = siamese.fit(
    [x_train_1, x_train_2],
    labels_train,
    validation_data=([x_val_1, x_val_2], labels_val),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[nn_TensorBoard ,es , mc ]
)
#%%
results = siamese.evaluate([x_test_1, x_test_2], labels_test)
print("test loss, test acc:", results)


# Epoch 7/50
# 972/972 [==============================] - ETA: 0s - loss: 9.8846e-04 - accuracy: 0.9987
# Epoch 00007: val_accuracy did not improve from 0.99875
# 972/972 [==============================] - 449s 462ms/step - loss: 9.8846e-04 - accuracy: 0.9987 - val_loss: 0.0021 - val_accuracy: 0.9976
# Epoch 00007: early stopping



#####################
#####################
#%%
Alsaba = "/Volumes/Lexar/DataScience/employee-attendance-facial-recognition/experiment_file/img/Faces/Alsaba_faces_resize_img/alsaba1218.jpg"
Mohammed = "/Volumes/Lexar/DataScience/employee-attendance-facial-recognition/experiment_file/img/Faces/Mohammed_faces_resize_img/mohammed1_218.jpg"
Girl = "/Volumes/Lexar/Downloads/105_classes_pins_dataset/pins_Alexandra Daddario/Alexandra Daddario0_214.jpg"


#%%
img1_path = Girl
image1 = np.array(Image.open(img1_path))
image1 = np.resize(image1, (1 ,100, 100, 3))
image1 = image1.astype('float32')
image1 /= 255


img2_path = Girl
image2 = np.array(Image.open(img2_path))
image2 = np.resize(image2, (1 ,100, 100, 3))
image2 = image2.astype('float32')
image2 /= 255


prediction = siamese.predict([image1, image2])
print(prediction)







