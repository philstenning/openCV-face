import cv2
import numpy as np

cap = cv2.VideoCapture('http://192.168.55.6:8080/video')
fgbg = cv2.createBackgroundSubtractorMOG2()


def nothing(x):
    pass


cv2.namedWindow('original')
cv2.createTrackbar('LOW', 'original', 0, 255, nothing)
cv2.setTrackbarPos('LOW', 'original', 180)

while True:
    contrast_low = cv2.getTrackbarPos('LOW', 'original')
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    ret, thresh1 = cv2.threshold(fgmask, contrast_low, 255, cv2.THRESH_BINARY)

    res = cv2.bitwise_and(frame, frame, mask=thresh1)

    cv2.imshow('original', frame)
    cv2.imshow('thresh1', thresh1)
    cv2.imshow('fgbgd', res)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
