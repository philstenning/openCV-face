from flask import Flask, render_template, Response
from camera import VideoCamera
# from camera_1 import VideoCamera1
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# def gen_2(camera_1):
#     while True:
#         frame = camera_1.get_frame()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/video_feed_1')
# def video_feed_1():
#     return Response(gen(VideoCamera1()),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)
