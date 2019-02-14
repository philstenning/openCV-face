import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# set res of camera
cap.set(3, 320)
cap.set(4, 240)
cv2.namedWindow('frame')
cv2.namedWindow('crop_img2')
cv2.namedWindow('crop2')
cv2.useOptimized()

row = [50, 400, 650]
col = [50, 265]
cv2.moveWindow('frame', row[0], col[0])
cv2.moveWindow('crop_img2', row[1], col[1])
cv2.moveWindow('crop2', row[1], col[0])


def nothing(x):
    pass


cv2.createTrackbar('LOW', 'frame', 0, 255, nothing)
cv2.setTrackbarPos('LOW', 'frame', 175)

cv2.createTrackbar('HIGH', 'frame', 0, 255, nothing)
cv2.setTrackbarPos('HIGH', 'frame', 255)


# def create_crop(npy_frame,x,y,w,h):

#         crop_img2 = npy_frame[120:200, 0:320]
#         gray = cv2.cvtColor(crop_img2, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     ret, processed_cropped_image = cv2.threshold(
#         blur, low, high, cv2.THRESH_BINARY)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # get current positions of four trackbars
    low = cv2.getTrackbarPos('LOW', 'frame')
    high = cv2.getTrackbarPos('HIGH', 'frame')

    #####################################
    # CROPPED BOX

    # create the bottom crop box
    crop_img2 = frame[120:200, 0:320]

    # add the filters
    gray = cv2.cvtColor(crop_img2, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, processed_cropped_image = cv2.threshold(
        blur, low, high, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(processed_cropped_image, cv2.MORPH_OPEN, kernel)

    # create box at top and bottom so we get a square to find later
    cv2.rectangle(opening, (0, 0), (320, 10), (0, 0, 0), -1)
    cv2.rectangle(opening, (0, 70), (320, 120), (0, 0, 0), -1)

    im2, contours, hierarchy = cv2.findContours(
        opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:

        # use the first contor only
        cnt = contours[0]

        area = cv2.contourArea(cnt)
        # print('\n\narea\n')
        # print(area)

        # add content area as rectangle to color image, you dont see it on the mask
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(crop_img2, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # add upper text to image
        text = '{} {} {} {}'.format(x, y, w, h)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(crop_img2, text, (0, 12), font,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

        # add center point to image
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        cv2.circle(crop_img2, center, 3, (67, 95, 0), 2)

        # write center data to screen
        img_center = 160
        res = -(img_center - int(x))

        text = 'center: {}'.format(res)
        cv2.putText(crop_img2, text, (0, 70), font,
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        ##################################

    cv2.imshow('frame', frame)

    # cv2.imshow('crop1', crop_img)
    cv2.imshow('crop2', opening)
    cv2.imshow('crop_img2', crop_img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
