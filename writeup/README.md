# Employee attendance using facial recognition

# Abstract:
In our project, we try to record employee attendace using facial biometric. Loopholes can be used to record attendance; for example, pointing a picture in front of the camera, that ultimately will count the person attended. In order to remedy that, we try to create liveness detector that is capable of identifying real from fake/spoofed faces. Two methods were used, first was using LivenessNet from PyImageSerch, and second was implementing a solution from a paper where it utilizes depth of two dimensional phases. Moreover, facial recognition is implemented using Siames neural network model.


# Liveness using LivenessNet:

## * Approach
In our approach, we implement a Single shot detector which OpenCV’s face detector is based on. At first, we used a pretrained model called ResNet 101. This serialized caffee model is supplied by PyImageSearch which accepts an input of 300x300.

Our model dataset is splitted into two classes; real and “fake/spoofed”. Images for real class were captured from an iPhone 7 Plus recorded video. On the other hand, fake/spoofed images were recorded on Logitech HD Pro Webcam C920 while holding the iPhone in front of the camera playing the real video. Each recording was at 30 frames per second. Populated images for both classes amounted to 1800 images for each class, totalling 3600 (i.e frame by frame image capturing)

## * Network architecture:

We used pre trained convolutional neural network (Livenessnet) with some tweaks of our own. 4 Conv2D were constructed with 16, 16, 32, 32 filters respectively, each followed by “Relu” activation. Batch normalization was used after each activation that standardized inputs to a layer for each mini-batch. Pooled feature map size to 2x2. 

# Image wrapping using two fast dimensional phase:

## * Approach
In this approach, features of images were extracted based on their hue, saturation and value. Red, green, blue channels intensified as an output for each as a seperate image. This would help identify the depth of each image and able to detect images from electronic device or any solid medium.

## * Network architecture:
CONV => RELU => POOL
CONV => RELU => POOL 
CONV => RELU => GLOBAL POOL

Dense(10) => RELU => Dense(240) => RELU => Dense(200) => RELU => Dense(2) => SOFTMAX



# Tools:
* Numpy
* OpenCV
* Tensorflow
* Keras
* Matplotlib
* Twilio

# Communication:
Along with slides, demo was provided during presentation.

# References

[Haar cascades](https://www.pyimagesearch.com/2021/04/05/opencv-face-detection-with-haar-cascades/)

[Face detection tips, suggestions, and best practices](https://www.pyimagesearch.com/2021/04/26/face-detection-tips-suggestions-and-best-practices/)

[Fast two-dimensional phase-unwrapping algorithm based on sorting by reliability following a noncontinuous path](https://www.researchgate.net/publication/233811917_Fast_two-dimensional_phase-unwrapping_algorithm_based_on_sorting_by_reliability_following_a_noncontinuous_path)

[Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

