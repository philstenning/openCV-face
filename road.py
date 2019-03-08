import numpy as np
import cv2
import time
from simple_pid import PID

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('http://192.168.55.6:8080/video')

# set res of camera
settings = {
    "window_x": 320,
    "window_y": 240,
    "crop_window_height": 80,
    "contrast_high": 255,
    "contrast_low": 160,
    "contrast_auto": True,
    "debug_mode": True,
    "display_on_screen": False,
    "follow_nearest_to_center": True

}
data = np.zeros(4, dtype=int)

# contrast_pid = PID(1, .1, .1, setpoint=1)
print(cv2.useOptimized())

# do not remove used in the trackbar control.


def nothing(x):
    pass


cap.set(3, settings['window_x'])
cap.set(4, settings['window_y'])
time.sleep(1)

# create variables from settings needed at runtime.
contrast_low = settings['contrast_low']
box_2_position = settings['window_y'] - 80

# the font used in the output frame window.
font = cv2.FONT_HERSHEY_PLAIN

# variables for the frame counter
frame_counter = 0
start_time = time.time()
fps = 0

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
    global contrast_low
    print('contrast low: {}'.format(contrast_low))
    contrast_low = contrast_low + int(new_value)

    if settings['debug_mode'] is True:
        cv2.setTrackbarPos('LOW', 'frame', contrast_low)
        print('box_2_position_trackbar: {}'.format(contrast_low))


def create_crop_box(position):
    b = position + 80
    c = 0
    d = 360
    center = 0
    contore_count = 0

    cropped_frame = frame[position:b, c:d]

    # add the filters
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # convert to a binary filter
    ret, processed_cropped_image = cv2.threshold(
        blur, contrast_low, settings['contrast_high'], cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    crop_color = cv2.morphologyEx(
        processed_cropped_image, cv2.MORPH_OPEN, kernel)

    # create box at top and bottom so we get a  nice square to process against.
    cv2.rectangle(crop_color, (0, 0), (d, 10), (0, 0, 0), -1)
    cv2.rectangle(crop_color, (0, 70), (d, b), (0, 0, 0), -1)
    im2, contours, hierarchy = cv2.findContours(
        crop_color, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contore_count = len(contours)

    if 1 <= contore_count <= 2:

        ###################
        # TODO: find largest contore and  follow it
        # TODO: T junction
        ###################

        ##################################################
        # find the contore nearest to center and return it
        ##################################################

        # only if there is more than two contours
        if contore_count >= 2:
            if settings['follow_nearest_to_center']:
                center_0 = find_center_of_contour(contours[0])
                center_1 = find_center_of_contour(contours[1])

                # remove negative numbers
                width = settings['window_x']/2
                c_0 = 0
                if center_0 < width:
                    c_0 = width-center_0
                else:
                    c_0 = center_0 - width

                # find the nearest to the center.
                c_1 = 0
                if center_1 < width:
                    c_1 = width-center_1
                else:
                    c_1 = center_1 - width

                # and draw the color rectangles around them.
                if c_0 <= c_1:
                    center = draw_rectangles(
                        contours[0], cropped_frame, center, 'green')
                    draw_rectangles(
                        contours[1], cropped_frame, center)
                else:

                    draw_rectangles(
                        contours[0], cropped_frame, center)
                    center = draw_rectangles(
                        contours[1], cropped_frame, center, 'green')

        # we only have one so it's green
        else:
            center = draw_rectangles(
                contours[0], cropped_frame, center, 'green')

        # area = cv2.contourArea(cnt)
        # print('\n\narea\n')
        # print(area)
        ##################################

    # we have too many contours so adjust the contrast
    elif len(contours) >= 3:
        set_contrast_low(5)

    else:
       # we have no contours pull it down a lot
       # then let it increese slowly backup
        set_contrast_low(-30)
    return crop_color, center, contore_count


def find_center_of_contour(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    img_center = 160
    center = str(-(img_center - int(x)))
    return int(x)

# draws the bounding box around the contore.
# returns the center cords


def draw_rectangles(cnt, cropped_frame, center, color='red'):

    r_x, r_y, r_w, r_h = cv2.boundingRect(cnt)
    if color == 'green':
        cv2.rectangle(cropped_frame, (r_x, r_y),
                      (r_x+r_w, r_y+r_h), (0, 255, 0), 2)
    else:
        cv2.rectangle(cropped_frame, (r_x, r_y),
                      (r_x+r_w, r_y+r_h), (0, 0, 255), 2)

    # add center point to image
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    cv2.circle(cropped_frame, center, 1, (67, 95, 0), 2)

    # write center data to screen
    img_center = 160
    res = str(-(img_center - int(x)))
    center_x, center_y = center
    cv2.putText(cropped_frame, res, (center_x-15,  center_y+20), font,
                1, (255, 255, 255), 2, cv2.LINE_AA)
    return center


def print_fps(frame, fps):
    text = 'fps:{}'.format(fps)
    cv2.putText(frame, text, (5, 15), font,
                1, (255, 255, 255), 2, cv2.LINE_AA)


# read 45 frames and throw them away
# just lets the camera settle before we
# start to do any work with it
for x in range(45):
    ret, frame = cap.read()


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_counter = frame_counter + 1

    if settings['debug_mode']:
        # get current positions the trackbars
        box_2_position = cv2.getTrackbarPos('box_2_position_trackbar', 'frame')
        contrast_low = cv2.getTrackbarPos('LOW', 'frame')

    # create a crop boxes to work from
    crop_mask_1, center_1, contore_count = create_crop_box(
        0)

    # get current positions of four trackbars
    # the settings may have been updated in the previous call.
    if 1 <= contore_count <= 2:
        low = cv2.getTrackbarPos('LOW', 'frame')
        crop_mask_2, center_2, contore_count_2 = create_crop_box(
            box_2_position)
        if settings['display_on_screen']:
            cv2.imshow('crop_mask_2', crop_mask_2)

    ######################
        # draw line between the two points.
        try:
            x_1, y_1 = center_1
            x_2, y_2 = center_2
            # cv2.line(frame, center_1, (x_2, y_2+box_2_position),
            #          (0, 255, 255), 1)
            c_x = int(40 + (box_2_position/2))
            c_y = 0
            if x_1 >= x_2:
                c_y = x_1 - x_2
                c_y = int(x_2 + (c_y/2))
            else:
                c_y = x_2 - x_1
                c_y = int(x_1 + (c_y/2))

            cv2.circle(frame, (c_y, c_x), 10, (0, 255, 0), 2)
            data[0] = x_1
            data[1] = c_y
            data[2] = x_2
            print(data)
        except:
            print('someting bad happened')
    ##################################

    # drive the bot
    # TODO FROM HERE

    ##################################
    if (time.time() - start_time) >= 1:
        fps = frame_counter
        start_time = time.time()
        frame_counter = 0
    print_fps(frame, fps)

    if settings['display_on_screen']:
        cv2.imshow('frame', frame)
        cv2.imshow('crop_mask_1', crop_mask_1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
