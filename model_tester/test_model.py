#https://pysource.com/2023/03/28/object-detection-with-yolo-v8-on-mac-m1-opencv-with-python-tutorial
import cv2
from ultralytics import YOLO
import numpy as np

MODEL_PATH = "weights/bestOld.pt"

VIDEO_PATH = "videos/various.mp4"

if VIDEO_PATH is not None:
    cap = cv2.VideoCapture(VIDEO_PATH)
else:
    cap = cv2.VideoCapture(0)

model = YOLO(MODEL_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame[200:840, 500:1140]

    results = model.predict(frame, device="mps")
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
        cv2.putText(frame, result.names[cls], (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)

    cv2.imshow("Img", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
