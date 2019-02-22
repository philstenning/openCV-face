import numpy as np
import cv2
import time

your_ip_address = '192.168.55.6'
# cap = cv2.VideoCapture(0) # use this for your webcam
# cap = cv2.VideoCapture('http://{}/video'.format(your_ip_address))
cap = cv2.VideoCapture('http://192.168.55.6:8080/video')
font = cv2.FONT_HERSHEY_PLAIN


def print_fps(frame, fps):
    text = 'fps:{}'.format(fps)
    cv2.putText(frame, text, (5, 15), font,
                1, (255, 255, 255), 2, cv2.LINE_AA)


frame_counter = 0
start_time = time.time()
fps = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # increment the frame counter
    frame_counter = frame_counter + 1
    if (time.time() - start_time) >= 1:
        fps = frame_counter
        start_time = time.time()
        frame_counter = 0
    print_fps(frame, fps)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
