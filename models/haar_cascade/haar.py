from imutils.video import VideoStream # video stream
import argparse # command line parser
import imutils # resize images
import time 
import cv2 # image processing

# argument parser, by default if not changed, it will take residing xml file. 
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", type=str,
	default="haarcascade_frontalface_default.xml",
	help="path to haar cascade face detector")
args = vars(ap.parse_args())

# load the haar cascade face detector from
print("><((('> LOADING FACE DETECTOR...")
detector = cv2.CascadeClassifier(args["cascade"])

# intialize video stream
print("><((('> STARTING VIDEO STREAM...")
vs = VideoStream(src=0).start()
time.sleep(2.0) # camera warm up

# loop over the frames from the video stream
while True:
	# grab the frame from the video stream, resize it, and convert it
	# to grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# perform face detection
	rects = detector.detectMultiScale(gray, scaleFactor=1.05,
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	# loop over the bounding boxes
	for (x, y, w, h) in rects:
		# bounding box
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

	# output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# exit loop by pressing 'q'
	if key == ord("q"):
		break

# housekeeping
cv2.destroyAllWindows()
vs.stop()