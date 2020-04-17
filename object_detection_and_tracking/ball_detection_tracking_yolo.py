import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import time

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.', default='soccer-ball.mp4')
args = parser.parse_args()

# Load names of classes
classesFile = "yolo/coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolo/yolov3.cfg"
modelWeights = "yolo/yolov3.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box if it detects a ball.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.putText(frame, "Detection", (left, top-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv.LINE_AA)
    # cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 0, 0), cv.FILLED)
    # cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        classId = classIds[i]
        if classId == 32:
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
            return (left, top, width, height)

# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

detection_fail = False
tracking_fail = False

while cv.waitKey(1) < 0:

    t1 = time.time()

    # get frame from the video
    hasFrame, frame = cap.read()

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(1000)
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    detections = postprocess(frame, outs)
    if detections is None and tracking_fail:
        detection_fail = True
        cv.putText(frame, "Lost track of the ball and not able to detect it!", (0, 85), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
    elif detections is None:
        detection_fail = True
        cv.putText(frame, "Not able to detect the ball!", (0, 85), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
    else:
        detection_fail = False

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'YOLO model inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 55), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

    t2 = time.time()
    fps = 1/(t2-t1)
    cv.putText(frame, f'FPS: {fps:.2f}', (0, 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.uint8))

    cv.imshow(winName, frame)
    k = cv.waitKey(100)
    if k == 27:
        break

    if not detection_fail:
        tracker = cv.TrackerKCF_create()
        # Initialize tracker with first frame and bounding box
        t1 = time.time()
        ok = tracker.init(frame, tuple(detections))

        while True:
            t1 = time.time()
            hasFrame, frame = cap.read()
            # Update tracker
            ok, bbox = tracker.update(frame)
            if ok:
                tracking_fail = False
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv.putText(frame, "Tracking", (p1[0], p1[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)
                cv.rectangle(frame, p1, p2, (0, 255, 0), cv.FONT_HERSHEY_DUPLEX, 4)
                cv.putText(frame, 'KCF Tracker', (0, 55), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
            else:
                tracking_fail = True
                break
            t2 = time.time()
            fps = 1/(t2-t1)
            cv.putText(frame, f'FPS: {fps:.2f}', (0, 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
            cv.imshow(winName, frame)
            vid_writer.write(frame.astype(np.uint8))
            k = cv.waitKey(1)
            if k == 27:
                break