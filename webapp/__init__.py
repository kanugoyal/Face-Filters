from flask import Flask, render_template, request
from flask.wrappers import Request, Response
import cv2
from .face_filter import face_filter

camera = cv2.VideoCapture(0)


def create_app():

    def gen_frames(val):  # generate frame by frame from camera
        while True:
            # Capture frame-by-frame
            success, frame = camera.read()
            if not success:
                break
            else:
                if val!=0:
                    frame = face_filter(frame, val)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    app=Flask(__name__)
    @app.route('/')
    def home():
        
        return render_template("index.html")

    @app.route('/video_feed', methods = ['POST', 'GET'])
    def video_feed():
        val = 0
        if request.method == 'POST':
            if request.form['img'] == 'Swag Glasses':
                val = 1
            if request.form['img'] == '3D Glasses':
                val = 2
            if request.form['img'] == 'Eyes':
                val = 3
            if request.form['img'] == 'Glasses':
                val = 4
            if request.form['img'] == 'Spiderman':
                val = 5
            if request.form['img'] == 'Ironman':
                val = 6
            
            
        return Response(gen_frames(val),mimetype='multipart/x-mixed-replace; boundary=frame')
       

    return app