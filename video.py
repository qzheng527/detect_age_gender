from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data import inputs
import multiprocessing
import numpy as np
import dlib
import tensorflow as tf
import cv2
import pyrealsense2 as rs
import time
FACE_PAD = 50

import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# realsense pipeline
pipeline = rs.pipeline()

def sub_image(img, x, y, w, h):
    upper_cut = [min(img.shape[0], y + h + FACE_PAD), min(img.shape[1], x + w + FACE_PAD)]
    lower_cut = [max(y - FACE_PAD, 0), max(x - FACE_PAD, 0)]
    roi_color = img[lower_cut[0]:upper_cut[0], lower_cut[1]:upper_cut[1]]
    return roi_color

def draw_rect(img, x, y, w, h):
    upper_cut = [min(img.shape[0], y + h + FACE_PAD), min(img.shape[1], x + w + FACE_PAD)]
    lower_cut = [max(y - FACE_PAD, 0), max(x - FACE_PAD, 0)]
    cv2.rectangle(img, (lower_cut[1], lower_cut[0]), (upper_cut[1], upper_cut[0]), (255, 0, 0), 2)


def draw_font(img, x, y, w, h, gender, age):
    font = cv2.FONT_HERSHEY_SIMPLEX
    upper_cut = [min(img.shape[0], y + h + FACE_PAD), min(img.shape[1], x + w + FACE_PAD)]
    lower_cut = [max(y - FACE_PAD, 0), max(x - FACE_PAD, 0)]
    text = gender + ' ' + str(int(age))
    print(text)
    cv2.putText(img, text, (lower_cut[1] + 32, lower_cut[0] + 32), font, 1, (255, 0, 0), 2, cv2.LINE_AA)


def display_video(mode, q, age_q, gender_q):
    if mode == 1:
        # start realsense camera
        # Configure depth and color streams
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)
    else:
        # Get a reference to webcam #0 (the default one)
        video_capture = cv2.VideoCapture(0)

        # start usb video and grab one frame
        ret, frame = video_capture.read()

    detector = dlib.get_frontal_face_detector()


    print('display started')
    while True:
        if mode == 1:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            # depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            img = np.asanyarray(color_frame.get_data())
        else:
            ret, img = video_capture.read()

        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(img, 1)
        files = []
        bb = []
        for (i, rect) in enumerate(faces):
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y
            bb.append((x, y, w, h))
            if q.full() is not True:
                files.append(sub_image(img, x, y, w, h))

        if len(files) > 0:
            print('%d faces detected' % len(files))
            q.put(files)

        guess_ages = []
        guess_genders = []
        draw_guess = False

        if gender_q.empty() is not True and age_q.empty() is not True:
            guess_genders = gender_q.get_nowait()
            for i, gender in enumerate(guess_genders):
                print('gender %d is %s' % (i+1, gender))
            guess_ages = age_q.get_nowait()
            for i, age in enumerate(guess_ages):
                print('age %d = %d' % (i+1, age))

        if len(bb) == len(guess_ages) and len(bb) == len(guess_genders):
            draw_guess = True

        for i, (x, y, w, h) in enumerate(bb):
            draw_rect(img, x, y, w, h)
            if draw_guess is True:
                draw_font(img, x, y, w, h, guess_genders[i], guess_ages[i])

        # Display the resulting image
        cv2.imshow('Video', img)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Stop streaming
    if mode == 1:
        pipeline.stop()
    else:
        video_capture.release()
    cv2.destroyAllWindows()
