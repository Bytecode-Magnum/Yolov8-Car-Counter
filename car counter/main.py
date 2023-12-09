from ultralytics import YOLO
import cv2
import math
import  numpy as np
import cvzone
from sort import *
model=YOLO('./weights/yolov8l.pt')
cap=cv2.VideoCapture('./testing data/cars_on_highway (1080p).mp4')
classes=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
         'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
         'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
         'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
         'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
         'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
         'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush']

map=cv2.imread('./mapped.png')
total=[]
tracker=Sort(min_hits=3,max_age=20,iou_threshold=0.3)
while True:
    rat,frame=cap.read()
    mapped_region = cv2.bitwise_and(frame, map)
    results = model.predict(mapped_region)
    detections = np.empty((0, 5))
    for result in results:
        bboxes = result.boxes
        for box in bboxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w = x2 - x1
            h = y2 - y1
            # print('type is',type(x1))
            conf = math.ceil(box.conf[0] * 100) / 100
            class_id = int(box.cls[0])
            current_class = classes[class_id]
            if current_class == 'car' or current_class == 'truck' or current_class == 'bus':

                cur_arr = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, cur_arr))
    tracker_result = tracker.update(detections)
    cv2.line(frame, (569, 621), (1520, 608), (255, 0, 0), 2)
    for result in tracker_result:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # print(result)
        w, h = x2 - x1, y2 - y1
        # print(type(x1))
        cvzone.cornerRect(frame, (x1, y1, w, h), l=4, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(frame, f'{int(id)}', (max(0, x1), max(35, y1)), 2, 3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        if (569 < cx < 1520) & ((621 - 20) < cy < 608 + 20):
            if total.count(id)==0:
                total.append(id)
                cv2.line(frame, (569, 621), (1520, 608), (255, 0,255), 2)
    cvzone.putTextRect(frame, f' total cars: {len(total)}', (max(0,918),max(0,30)), 2,offset=10)
    cv2.imshow('Frame',frame)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
    # cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
