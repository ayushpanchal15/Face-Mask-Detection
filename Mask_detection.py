# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
def mask_prediction(frame, fNet, mNet):
	(h, w) = frame.shape[:2]
	binary_large_object = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))
	fNet.setInput(binary_large_object)
	detect = fNet.forward()
	print(detect.shape)
	faces = []
	locations = []
	predictions = []
	for i in range(0, detect.shape[2]):
		probability = detect[0, 0, i, 2]
		if probability > 0.5:
			boundingbox = detect[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = boundingbox.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			faces.append(face)
			locations.append((startX, startY, endX, endY))
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		predictions = mNet.predict(faces, batch_size=49)
	return (locations, predictions)

txtPath = r"D:\CODE\Face-Mask-Detection-master\face_detector\deploy.prototxt"
weightPath = r"D:\CODE\Face-Mask-Detection-master\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
fNet = cv2.dnn.readNet(txtPath, weightPath)


mNet = load_model("mask_detector.model")


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()


while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	(locations, predictions) = mask_prediction(frame, fNet, mNet)
	for (boundingbox, pred) in zip(locations, predictions):
		(startX, startY, endX, endY) = boundingbox     
		(mask, withoutMask) = pred
		lbl = "Mask" if mask > withoutMask else "No Mask"
		clr = (255, 0, 0) if lbl == "Mask" else (0, 0, 255)
		cv2.putText(frame, lbl, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), clr, 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
vs.stop()