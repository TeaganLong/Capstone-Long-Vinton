import numpy as np
import argparse
import cv2

# Structuring the argument parse for the model
parser = argparse.ArgumentParser(
    description="Script to run MobileNet-SSD object detection network"
)
parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                    help="Path to test network file: "
                    "MobileNetSSD_deploy.prototxt for Caffe model or "
                    )
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                    help="Path to weights: "
                    "MobileNetSSD_deploy.caffemodel for Caffe model or ")
parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
args = parser.parse_args()

# Labels for the model
classNames = {15: 'person'}

# Opening the video capture
cam = cv2.VideoCapture(0)

# Loading the model
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

while True:
    # Capturing frame-by-frame
    ret, img = cam.read()
    img_resized = cv2.resize(img, (300, 300))

    # Dimensions for input images
    blob = cv2.dnn.blobFromImage(img_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    # Setting network to input blob
    net.setInput(blob)
    # Network Prediction(s)
    detections = net.forward()

    # Size of the resized frame
    cols = img_resized.shape[1]
    rows = img_resized.shape[0]

    # Obtaining class/location of detected object(s)
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args.thr:
            class_id = int(detections[0, 0, i, 1])

            # Object location(s)
            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop = int(detections[0, 0, i, 5] * cols)
            yRightTop = int(detections[0, 0, i, 6] * rows)

            # Scale factor to original size of frame
            heightFactor = img.shape[0]/300.0
            widthFactor = img.shape[1]/300.0

            # Scaling object detection to frame
            xLeftBottom = int(widthFactor * xLeftBottom)
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop = int(widthFactor * xRightTop)
            yRightTop = int(heightFactor * yRightTop)

            # Drawing the location of the object(s)
            cv2.rectangle(img, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0))

            # Drawing the label and confidence (% match)
            if class_id in classNames:
                label = classNames[class_id] + ": " + str(confidence)
                labelSize, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                yLeftBottom = max(yLeftBottom, labelSize[1])

                # Printing label name and confidence
                cv2.rectangle(img, (xLeftBottom, yLeftBottom - labelSize[1]),
                              (xLeftBottom + labelSize[0], yLeftBottom + baseline),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(img, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow("Object Detection", img)
    if cv2.waitKey(1) >= 0:
        break

cam.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
