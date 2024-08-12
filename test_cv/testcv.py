import cv2
import time
import numpy as np

def testcv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # print("Timestamp for hsv variable: ", time.time() - self.prev_time)
    mask = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])) + \
        cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    # print("Timestamp for mask variable: ", time.time() - self.prev_time)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("Timestamp for contours variable: ", time.time() - self.prev_time)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # print("Timestamp for largest_contour variable: ", time.time() - self.prev_time)
        if cv2.contourArea(largest_contour) > 0:
            x, y, w, h = cv2.boundingRect(largest_contour)
            # print("Timestamp for return: ", time.time() - self.prev_time)
            return {"status": True, "xmin": x, "xmax": x + w, "ymin": y, "ymax": y + h}, frame
        

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            pre_time = time.time()
            result = testcv(frame)
            print("Time taken: ", time.time() - pre_time)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()