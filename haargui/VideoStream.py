import cv2
import threading
import time

class VideoStream:
	def __init__(self, src=0, name='VideoStream'):
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		self.name = name
		self.stopped = False

	def start(self):
		# start thread to read frames from video stream
		t = threading.Thread(target=self.update, name=self.name, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		# keep looping infinitely until thread is stopped
		while True:
			if self.stopped:
				return
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		return self.frame

	def stop(self):
		self.stopped=True