from VideoStream import VideoStream
from FaceFilters import FaceFilters
from faceTk import GUIFace
import time

filters = ['glasses.png', 'eyes.png','eyelasses.png','3dglasses.png', 'swag.png', 'cat.png', \
        'monkey.png', 'rabbit.png','moustache.png', 'moustache1.png', 'ironman.png','spiderman.png','batman.png', 'capAmerica.png',]

vs = VideoStream(0).start()
fc = FaceFilters(filters)
time.sleep(2.0)
gui = GUIFace(vs,fc,'output')