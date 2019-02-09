import numpy as np
import cv2
import json
cap = cv2.VideoCapture(0)

img = np.zeros((1, 500, 3), np.uint8)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.namedWindow('mask')
cv2.namedWindow('res')
cv2.namedWindow('frame')
cv2.namedWindow('contors')


# set the initial color to calibrate
current_color = 'blue'


def nothing(x):
    pass


cv2.createTrackbar('LOW H', 'image', 0, 255, nothing)
cv2.createTrackbar('LOW S', 'image', 0, 255, nothing)
cv2.createTrackbar('LOW V', 'image', 0, 255, nothing)
cv2.createTrackbar('UP H', 'image', 0, 255, nothing)
cv2.createTrackbar('UP S', 'image', 0, 255, nothing)
cv2.createTrackbar('UP V', 'image', 0, 255, nothing)


def load_data():
    with open('./colors/' + current_color + '.txt') as json_file:
        data = json.load(json_file)
        print(data)
        return data


def set_trackbar_positions(data):
    cv2.setTrackbarPos('LOW H', 'image', data['la'])
    cv2.setTrackbarPos('LOW S', 'image', data['lb'])
    cv2.setTrackbarPos('LOW V', 'image', data['lc'])
    cv2.setTrackbarPos('UP H', 'image', data['ua'])
    cv2.setTrackbarPos('UP S', 'image', data['ub'])
    cv2.setTrackbarPos('UP V', 'image', data['uc'])


def save_data(data):
    with open('./colors/' + current_color + ".txt", "w") as outfile:
        json.dump(data, outfile)


# get the initial data from file.
# data =
set_trackbar_positions(load_data())

row = [50, 790, 1500]
col = [50, 600]
cv2.moveWindow('image', row[0], col[0])
cv2.moveWindow('mask', row[1], col[0])
cv2.moveWindow('res', row[0], col[1])
cv2.moveWindow('frame', row[1], col[1])
cv2.moveWindow('contors', row[2], col[0])


while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # get current positions of four trackbars
    la = cv2.getTrackbarPos('LOW H', 'image')
    lb = cv2.getTrackbarPos('LOW S', 'image')
    lc = cv2.getTrackbarPos('LOW V', 'image')
    ua = cv2.getTrackbarPos('UP H', 'image')
    ub = cv2.getTrackbarPos('UP S', 'image')
    uc = cv2.getTrackbarPos('UP V', 'image')

    # define range of blue color in HSV 86,154,168 x 108 , 255 ,255
    lower_blue = np.array([la, lb, lc])
    upper_blue = np.array([ua, ub, uc])

    # blur = cv2.GaussianBlur(frame, (9, 9), 0)
    blur = cv2.blur(frame, (9, 9))
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(blur, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    im2, contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    # cv2.drawContours(blur, contours, -1, (0, 255, 0), 3)

    if len(contours) > 0:
        cnt = contours[0]
        # cv2.drawContours(blur, [cnt], 0, (0, 255, 0), 3)
        # M = cv2.moments(cnt)
        # area = cv2.contourArea(cnt)
        # print(area)

        # Draw a bounding box
        # rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # cv2.drawContours(blur, [box], 0, (0, 0, 255), 2)

        # Draw a circle.
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))

        radius_outside = int(radius)
        radius = 3
        cv2.circle(blur, center, radius, (0, 255, 0), 2)
        cv2.circle(blur, center, radius_outside, (0, 255, 0), 2)

    cv2.imshow('frame', blur)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    cv2.imshow('image', img)
    cv2.imshow('contors', im2)

    def get_data():
        return {
            'la': la,
            'lb': lb,
            'lc': lc,
            'ua': ua,
            'ub': ub,
            'uc': uc
        }

    def change_color(new_color):
        data = get_data()
        global current_color
        data = get_data()
        save_data(data)
        current_color = new_color
        set_trackbar_positions(load_data())

        print(new_color.upper())

    if cv2.waitKey(1) & 0xFF == ord('q'):  # wait for ESC key to exit
        data = get_data()
        save_data(data)

        break
    elif cv2.waitKey(10) & 0xFF == ord('r'):
        if current_color != 'red':
            change_color('red')
    elif cv2.waitKey(1) & 0xFF == ord('b'):
        if current_color != 'blue':
            change_color('blue')
    elif cv2.waitKey(1) & 0xFF == ord('g'):
        if current_color != 'green':
            change_color('green')

    elif cv2.waitKey(1) & 0xFF == ord('y'):
        if current_color != 'yellow':
            change_color('yellow')


cv2.destroyAllWindows()
