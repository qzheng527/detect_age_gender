from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
from data import inputs
import numpy as np
import tensorflow as tf
from model import select_model, get_checkpoint
from utils import *
import os
import json
import csv
import dlib
import cv2
import pyrealsense2 as rs
import sys
import multiprocessing
from video import display_video
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


RESIZE_FINAL = 227
RESIZE_AOI = 256
GENDER_LIST =['M','F']
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
AGE_LIST_MID = [1.0, 5.0, 10.0, 17.5, 28.5, 40.5, 50.5, 80.0]
MAX_BATCH_SZ = 128


def make_multi_crop_raw(img):
    logger.info('Running multi-cropped image')
    image = tf.image.resize_images(img, (RESIZE_AOI, RESIZE_AOI))
    crops = []
    h = RESIZE_AOI
    w = RESIZE_AOI
    hl = h - RESIZE_FINAL
    wl = w - RESIZE_FINAL

    crop = tf.image.resize_images(image, (RESIZE_FINAL, RESIZE_FINAL))
    crops.append(standardize_image(crop))
    crops.append(tf.image.flip_left_right(crop))

    corners = [(0, 0), (0, wl), (hl, 0), (hl, wl), (int(hl / 2), int(wl / 2))]
    for corner in corners:
        ch, cw = corner
        cropped = tf.image.crop_to_bounding_box(image, ch, cw, RESIZE_FINAL, RESIZE_FINAL)
        crops.append(standardize_image(cropped))
        flipped = tf.image.flip_left_right(cropped)
        crops.append(standardize_image(flipped))

    image_batch = tf.stack(crops)
    return image_batch

def classify_one_multi_crop(sess, label_list, softmax_output, images, image):
    try:
        image_batch = make_multi_crop_raw(image)
        logger.info('====')

        batch_results = sess.run(softmax_output, feed_dict={images:image_batch.eval()})
        output = batch_results[0]
        batch_sz = batch_results.shape[0]

        for i in range(1, batch_sz):
            output = output + batch_results[i]

        output /= batch_sz
        best = np.argmax(output)
        best_choice = (label_list[best], output[best])
        logger.info('Guess @ 1 %s, prob = %.2f' % best_choice)


        nlabels = len(label_list)
        if nlabels > 2:
            output[best] = 0
            second_best = np.argmax(output)
            second_choice = (label_list[second_best], output[second_best])
            logger.info('Guess @ 2 %s, prob = %.2f' % second_choice)
            # nlables > 2 means it is for age guess
            guess_age = (AGE_LIST_MID[best]*best_choice[1] + AGE_LIST_MID[second_best]*second_choice[1])/(best_choice[1] + second_choice[1])
            # logger.info('guess age = %d' % guess_age)
            return guess_age
        else:
            # nlables = 2 means it is for gender guess
            # logger.info('guess gender = %s' % best_choice[0])
            return best_choice[0]

    except Exception as e:
        print(e)
        print('Failed to run image crop ')
        return None



def guess_loop(class_type, model_type, model_dir, q, guess_q):
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        label_list = AGE_LIST if class_type == 'age' else GENDER_LIST
        nlabels = len(label_list)

        model_fn = select_model(model_type)

        with tf.device('/cpu:0'):
            images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
            logits = model_fn(nlabels, images, 1, False)

            init = tf.global_variables_initializer()

            checkpoint_path = '%s' % (model_dir)

            model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, None, 'checkpoint')

            saver = tf.train.Saver()
            saver.restore(sess, model_checkpoint_path)

            softmax_output = tf.nn.softmax(logits)


            while True:
                image_files = q.get()
                guess_result = []

                for image in image_files:
                    guess = classify_one_multi_crop(sess, label_list, softmax_output, images, image)
                    if guess is not None:
                        guess_result.append(guess)

                if guess_q.full() is not True:
                    guess_q.put(guess_result)



if __name__ == '__main__':
    if sys.argv[1] == 'rs':
        mode = 1
    else:
        mode = 0

    # video_loop(mode)
    q = multiprocessing.Queue(4)
    age_q = multiprocessing.Queue(2)
    gender_q = multiprocessing.Queue(2)

    p1 = multiprocessing.Process(target=display_video, args=(mode, q, age_q, gender_q))
    p2 = multiprocessing.Process(target=guess_loop, args=('age', 'inception', './checkpoints/age/22801', q, age_q))
    p3 = multiprocessing.Process(target=guess_loop, args=('gender', 'inception', './checkpoints/gender/21936', q, gender_q))
    p1.daemon = True
    p2.daemon = True
    p3.daemon = True
    p3.start()
    p2.start()
    p1.start()
    p1.join()