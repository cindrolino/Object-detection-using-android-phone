import numpy as np
import cv2
import urllib.request

# Load the pre-trained model for object detection
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# Define the classes of objects we want to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Set the minimum confidence level for object detection
CONFIDENCE_THRESHOLD = 0.5

# Set the IP address and port of the IP Webcam server
url = "http://192.168.31.17:8080/shot.jpg"

while True:
    # Open the URL and read the image from the IP Webcam server
    img_resp = urllib.request.urlopen(url)
    img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(img_np, -1)

    # Convert the image to a blob (binary large object)
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)

    # Pass the blob through the network and get the detections
    net.setInput(blob)
    detections = net.forward()

    # Loop through the detections and draw boxes around the objects
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            class_id = int(detections[0, 0, i, 1])
            class_label = CLASSES[class_id]
            (startX, startY, endX, endY) = (detections[0, 0, i, 3:7] * [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).astype("int")
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(img, class_label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with object detection
    cv2.imshow('Object Detection', img)

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close all windows
cv2.destroyAllWindows()

