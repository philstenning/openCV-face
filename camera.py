import argparse
import numpy as np
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('-W', '--websocket_server')
parser.add_argument('-V', '--video_server')

args = parser.parse_args()

web_socket_server = "192.168.55.18:5001"
if (args.websocket_server):
    web_socket_server = args.websocket_server

video_server = "192.168.55.13:8081"
if (args.video_server):
    video_server = args.video_server

print('\n video feed url: {} \n websocket server url:{}'.format(
    video_server, web_socket_server))


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture("http://{}/".format(video_server))

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        image = cv2.medianBlur(image, 5)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, jpeg = cv2.imencode('.jpg', gray)

        return jpeg.tobytes()
