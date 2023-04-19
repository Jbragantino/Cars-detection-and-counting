import cv2
import math
import time

car_classifier = cv2.CascadeClassifier('XML\cars.xml')

cap = cv2.VideoCapture('Vídeo/video.avi')

center_list = []

def save_center(center_x, center_y, threshold = 10):
    for i in range(len(center_list)):
        if math.dist(center_list[i], [center_x, center_y]) < threshold:
            center_list[i] = (center_x, center_y)
            return
        
    center_list.append((center_x, center_y))


while cap.isOpened():
    
    # time.sleep(.05)

    ret, frame = cap.read()

    if (type(frame) == type(None)):
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    cars = car_classifier.detectMultiScale(blur, 1.1, 2)
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        center_x = (x+x+w)//2
        center_y = (y+y+h)//2
        center = cv2.circle(frame, (center_x, center_y), 0, (0, 0, 255), -1)

        save_center(center_x, center_y, 20)

        cv2.imshow('Cars detection', frame)

    if cv2.waitKey(1) == 13:
        break

print("Total de carros detectados:", len(center_list))