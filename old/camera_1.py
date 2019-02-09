import numpy as np
from imutils.video import VideoStream
import imutils
import cv2
import time

#  pip install "picamera[array]"


class VideoCamera1(object):
    def __init__(self):
        self.video = VideoStream(usePiCamera=True).start()
        time.sleep(2.0)

    def __del__(self):
        self.video.stop()

    def get_frame(self):
        success, image_raw = self.video.read()
        frame = imutils.resize(image_raw, width=400)
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        image = cv2.medianBlur(frame, 5)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # skeleton = imutils.skeletonize(gray, size=(3, 3))

        ret, jpeg = cv2.imencode('.jpg', gray)

        return jpeg.tobytes()
