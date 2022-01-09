In our project, we start with Haar cascades algorithm as a baseline; although it is an older set of algorithms that is used in object detection and specifically face detection; in addition, prone to false positive detections, it is prominent to understand how these set of algorithms work since it was our first project working with computer vision.

Primtively, we decided to lay out our issues and loopholes in our initial problem which is taking attendence of employees just by recognizing their faces. At first, users can fool the algorithm by pointing/displaying "fake/spoofed" picture of themselves at the camera and thus will count them as attended. In order to remedy that, we tried to detect the "liveliness" of the object. Next, we tried to be more fancy and capture emotions. For now we are limiting emotions for happy, sad and neutral.


## Liveness and spoofed

Using Haar cascades, we have been able to detect faces. The issue is, our accuracy is not stable; therefore, we had to manipulate the multi scale minimum neighbors parameters in order to achieve better detection.

![2022-01-09-225346_498x445_scrot](https://user-images.githubusercontent.com/49822946/148698453-33d9da68-162a-4c14-9e1b-3518ca09dedb.png)

Since this is a baseline we will discard its results. Next we will try to implement a shallow CNN with few parameters to enusre that it runs on our resource constrained device the Raspberry Pi.

## Emotions detection

In order to build a model able to detect emotional expressions, we gathered data from Kaggle and performed transfer learning by using mobileNet model. We ended up with accuracy score of 0.66 on the training, and 0.71 on the validation.

* Accuracy score:

![image](https://user-images.githubusercontent.com/89771282/148701087-c5d87f8e-14bf-43dd-96c1-460d56482679.png)

* Loss score:

![image](https://user-images.githubusercontent.com/89771282/148701072-0901b798-d0b1-4bbb-a8c5-842d4a9ead51.png)
