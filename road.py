import numpy as np
import cv2
import time
from simple_pid import PID

cap = cv2.VideoCapture(0)

# set res of camera
settings = {
    "window_x": 320,
    "window_y": 240,
    "crop_window_height": 80,
    "contrast_high": 255,
    "contrast_low": 150,
    "contrast_auto": True,
    "debug_mode": True

}

contrast_pid = PID(1, 0.1, 0.05, setpoint=1)
print(cv2.useOptimized())


def nothing(x):
    pass


cap.set(3, settings['window_x'])
cap.set(4, settings['window_y'])
time.sleep(1)

# create variables from settings needed at runtime.
contrast_low = settings['contrast_low']
box_2_position = settings['window_y'] - 80


if settings['debug_mode']:
    cv2.namedWindow('frame')
    cv2.namedWindow('crop_mask_1')
    cv2.namedWindow('crop_mask_2')
    row = [50, 400, 850]
    col = [50, 265]
    cv2.moveWindow('frame', row[0], col[0])
    cv2.moveWindow('crop_mask_1', row[1], col[0])
    cv2.moveWindow('crop_mask_2', row[1], col[1])
    cv2.createTrackbar('LOW', 'frame', 0, 255, nothing)
    cv2.setTrackbarPos('LOW', 'frame', contrast_low)

    # create track bars for box positions
    cv2.createTrackbar('box_2_position_trackbar', 'frame', 61,
                       box_2_position, nothing)
    cv2.setTrackbarPos('box_2_position_trackbar', 'frame', box_2_position)


def set_contrast_low(new_value):
    print(contrast_low)
    contrast_low = new_value
    cv2.setTrackbarPos('LOW', 'frame', new_value)
    print(new_value)


def create_crop_box(a):
    b = a + 80
    c = 0
    d = 360
    center = 0

    cropped_frame = frame[a:b, c:d]

    # add the filters
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, processed_cropped_image = cv2.threshold(
        blur, contrast_low, settings['contrast_high'], cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    crop_color = cv2.morphologyEx(
        processed_cropped_image, cv2.MORPH_OPEN, kernel)

    # create box at top and bottom so we get a square to find later
    cv2.rectangle(crop_color, (0, 0), (d, 10), (0, 0, 0), -1)
    cv2.rectangle(crop_color, (0, 70), (d, b), (0, 0, 0), -1)

    im2, contours, hierarchy = cv2.findContours(
        crop_color, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contore_count = len(contours)
    if contore_count == 1:

        # use the first contor
        cnt = contours[0]

        # add content area as rectangle to color image, you dont see it on the mask
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(cropped_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        font = cv2.FONT_HERSHEY_PLAIN
        # add center point to image
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        cv2.circle(cropped_frame, center, 1, (67, 95, 0), 2)

        # write center data to screen
        img_center = 160
        res = -(img_center - int(x))

        text = 'center: {}'.format(res)
        cv2.putText(cropped_frame, text, (0, 70), font,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

        # area = cv2.contourArea(cnt)
        # print('\n\narea\n')
        # print(area)
        ##################################
    elif len(contours) >= 1:

        if settings['contrast_auto']:
            pos = cv2.getTrackbarPos('LOW', 'frame') + 1
            cv2.setTrackbarPos('LOW', 'frame', pos)
            if settings["debug_mode"]:
                print('Contrast too low. {} '.format(pos))

    else:

        if settings['contrast_auto']:
            pos = cv2.getTrackbarPos('LOW', 'frame') - 1
            print('Contrast too high. {} '.format(pos))
            if settings["debug_mode"]:
                cv2.setTrackbarPos('LOW', 'frame', pos)

    return crop_color, center, contore_count


for x in range(45):
    ret, frame = cap.read()


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if settings['debug_mode']:
        # get current positions the trackbars
        box_2_position = cv2.getTrackbarPos('box_2_position_trackbar', 'frame')
        contrast_low = cv2.getTrackbarPos('LOW', 'frame')

    # create a crop boxes to work from
    crop_mask_1, center_1, contore_count = create_crop_box(
        0)

    # get current positions of four trackbars
    # the settings may have been updated in the previous call.
    if contore_count == 1:
        low = cv2.getTrackbarPos('LOW', 'frame')
        crop_mask_2, center_2, contore_count_2 = create_crop_box(
            box_2_position)
        cv2.imshow('crop_mask_2', crop_mask_2)

    ##################################

    cv2.imshow('frame', frame)
    cv2.imshow('crop_mask_1', crop_mask_1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
