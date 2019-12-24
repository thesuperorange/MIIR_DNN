
import numpy as np
import argparse
import cv2
import os
import time

CONFIDENCE_TH = 0.3


def SSD_detect(model_path,fo,dataset, foldername, filename, ch, mode_img, bbox_log):

	prototxt = model_path + "/MobileNetSSD_deploy.prototxt.txt"
	model = model_path + "/MobileNetSSD_deploy.caffemodel"

	# modelConfiguration = "yolov3-tiny.cfg"
	# modelBinary = "yolov3.weights"

	# initialize the list of class labels MobileNet SSD was trained to
	# detect, then generate a set of bounding box colors for each class

	labelsPath = os.path.sep.join([SSD_model, "object_detection_classes_coco.txt"])
	CLASSES = open(labelsPath).read().strip().split("\n")

	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

	# load our serialized model from disk
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe(prototxt, model)

	image_num = os.path.splitext(filename)[0]


	#print(foldername+"/"+filename)
	image = cv2.imread(foldername+"/"+filename)
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

	# print("[INFO] computing object detections...")
	net.setInput(blob)
	start = time.time()
	detections = net.forward()
	end = time.time()
	print("[INFO] took {:.6f} seconds".format(end - start))

	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > CONFIDENCE_TH:
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

		# display the prediction
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			#print("[INFO] {}".format(label))

			if mode_img:
				cv2.rectangle(image, (startX, startY), (endX, endY),COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(
					image, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2
				)

			if bbox_log:
				fo.write(
					str(ch)+","+image_num+","+str(startX)+"," + str(startY)+"," +
					str(endX)+"," + str(endY)+","+str(confidence)+","+CLASSES[idx]+"\n"
				)


	if mode_img:
		# show the output image
		# cv2.imshow("Output", image)
		output_folder = dataset+"_"+ch
		if not os.path.exists(output_folder):
			os.mkdir(output_folder)
		output_name = output_folder+"/"+filename
		print(output_name)
		cv2.imwrite(output_name, image)
		# cv2.waitKey(0)






