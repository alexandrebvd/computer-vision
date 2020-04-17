import cv2
import sys
import time

threshold = 0.5
inWidth = 320
inHeight = 320
mean = (127.5, 127.5, 127.5)

modelFile = "ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
configFile = "ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
classFile = "coco_class_labels.txt"
with open(classFile) as fi:
    labels = fi.read().split('\n')

source = './soccer-ball.mp4'
if len(sys.argv) > 1:
    source = sys.argv[1]

cap = cv2.VideoCapture(source)
ret, frame = cap.read()

net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))

tracker = cv2.TrackerMIL_create()

successive_detections = 0

while True:
    ret, frame = cap.read()
    rows = frame.shape[0]
    cols = frame.shape[1]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0/127.5, (inWidth, inHeight), mean, True, False))
    out = net.forward()

    for i in range(out.shape[2]):
        score = float(out[0, 0, i, 2])
        classId = int(out[0, 0, i, 1])

        x1 = int(out[0, 0, i, 3] * cols)
        y1 = int(out[0, 0, i, 4] * rows)
        x2 = int(out[0, 0, i, 5] * cols)
        y2 = int(out[0, 0, i, 6] * rows)

        if score > threshold and classId == 37:
            successive_detections += 1
            cv2.putText(frame, "Detection", ( x1, y1 - 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            # cv2.putText(frame, "{}, confidence = {:.3f}".format(labels[classId], score), ( x1, y1 - 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), cv2.FONT_HERSHEY_DUPLEX, 4)

            if successive_detections > 1:
                successive_detections = 0
                tracker = cv2.TrackerKCF_create()
                # Initialize tracker with first frame and bounding box
                t1 = time.time()
                ok = tracker.init(frame, (x1, y1, x2-x1, y2-y1))
                # print(x1,y1,x2,y2)
                # print(ok)
                # print(time.time() - t1)
                while True:
                    ret, frame = cap.read()
                    # Update tracker
                    ok, bbox = tracker.update(frame)
                    print(bbox)
                    if ok:
                        p1 = (int(bbox[0]), int(bbox[1]))
                        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                        # print(frame.shape)
                        # print(p1,p2)
                        cv2.putText(frame, "Tracking", ( p1[0], p1[1]-10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.rectangle(frame, p1, p2, (0, 255, 0), cv2.FONT_HERSHEY_DUPLEX, 4)
                    else:
                        break
                    cv2.imshow("OpenCV Tensorflow Object Detection Demo", frame)
                    k = cv2.waitKey(10)
                    if k == 27:
                        break

    cv2.imshow("OpenCV Tensorflow Object Detection Demo", frame)
    vid_writer.write(frame)
    k = cv2.waitKey(10)
    if k == 27:
        break

cv2.destroyAllWindows()
vid_writer.release()
