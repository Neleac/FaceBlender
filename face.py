import cv2
import numpy as np

from haar_cascade_classifier import HaarCascadeClassifier


RESOLUTION = (1280, 720)

# set up canvas
canvas = cv2.imread('art.jpg')
canvas = cv2.resize(canvas, RESOLUTION, interpolation=cv2.INTER_AREA)

# set up camera
cam = cv2.VideoCapture(0)
#cam = cv2.VideoCapture('audience.mp4')
if not cam.isOpened():
    print("Cannot open camera")
    exit()

# set up face detector
face_cascade_file = 'haarcascade_frontalface_default.xml'
face_detector = HaarCascadeClassifier(face_cascade_file)

# set up video writer
#frame_writer = cv2.VideoWriter('frame.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, RESOLUTION)
#canvas_writer = cv2.VideoWriter('canvas.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, RESOLUTION)

# live inference
while True:
    ret, frame = cam.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    faces = face_detector.classify(frame)
    for (x, y, w, h) in faces:
        face = frame[y : y + h, x : x + w]
        face = cv2.resize(face, (300, 300), interpolation=cv2.INTER_AREA)

        canvas[250:550, 450:750] = cv2.addWeighted(canvas[250:550, 450:750], 0.9, face, 0.1, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))

    cv2.imshow('frame', frame)
    cv2.imshow('canvas', canvas)
    #frame_writer.write(frame)
    #canvas_writer.write(canvas)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
#frame_writer.release()
#canvas_writer.release()
cv2.destroyAllWindows()
