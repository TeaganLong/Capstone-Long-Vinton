import argparse
import cv2

# Credit to djmv at https://github.com/djmv/MobilNet_SSD_opencv for
# instructions on how to use MobileNet for Object Detection

# Structuring the parser for the MobileNet model using the argparse library
arg_parser = argparse.ArgumentParser(description="This is a script that runs the MobileNet Object Detection")

# Setting the prototxt file for MobileNet
arg_parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt")

# Setting the Caffe model/weights for MobileNet
arg_parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel")

# Filtering out unnecessary predictions
arg_parser.add_argument("--thr", default=0.2, type=float)
args = arg_parser.parse_args()

# Filtering out MobileNet Labels to only include
# labels that identify a person
labels = {15: 'person'}

# Opening the camera feed
cam = cv2.VideoCapture(0)

# Loading the deep neural network with the above parser arguments
network = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

while True:
    # Capturing video feed frame-by-frame
    ret, img = cam.read()
    img_resized = cv2.resize(img, (300, 300))

    # Dimensions for input images
    blob = cv2.dnn.blobFromImage(img_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)

    # Setting network to input blob
    network.setInput(blob)

    # Network Prediction(s)
    predictions = network.forward()

    # Size of the resized frame (should be 300x300 pixels)
    columns = img_resized.shape[1]
    rows = img_resized.shape[0]

    # Obtaining the label/location of detected object(s)
    for i in range(predictions.shape[2]):
        confidence = predictions[0, 0, i, 2]  # How much does an image match other images in the dataset
        if confidence > args.thr:  # Filtering the predictions
            label_id = int(predictions[0, 0, i, 1])  # Labels

            # The object(s) location in each individual frame
            xLB = int(predictions[0, 0, i, 3] * columns)
            yLB = int(predictions[0, 0, i, 4] * rows)
            xRT = int(predictions[0, 0, i, 5] * columns)
            yRT = int(predictions[0, 0, i, 6] * rows)

            # Scale factor to original size of frame
            height = img.shape[0]/300.0
            width = img.shape[1]/300.0

            # Scaling object detection to size of current frame
            xLB = int(width * xLB)
            yLB = int(height * yLB)
            xRT = int(width * xRT)
            yRT = int(height * yRT)

            # Drawing the location(s) of the detected object(s)
            cv2.rectangle(img, (xLB, yLB), (xRT, yRT), (0, 255, 0))

            # Drawing the label and confidence of the prediction(s) (percent match)
            if label_id in labels:
                # Narrowing confidence to 5 decimal places, so it isn't a very small number
                label = labels[label_id] + ": " + str(round(confidence, 5))
                label_size, base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                yLB = max(yLB, label_size[1])

                # Displaying label/confidence of the detected object
                cv2.rectangle(img, (xLB, yLB - label_size[1]),
                              (xLB + label_size[0], yLB + base),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(img, label, (xLB, yLB),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow("Rescue Robot Demo (Press ESC to Exit)", img)
    if cv2.waitKey(1) >= 0:  # Press ESC key to close
        break

cam.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
