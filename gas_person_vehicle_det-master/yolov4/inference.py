#!/usr/bin/env python
# encoding: utf-8
'''
@author: 余智君
@license: (C) Copyright 2005-2025, 四川弘和集团.
@contact: yuzhijun@cwddd.com
@software:PyCharm2019.3
@file: inference.py
@time: 2020/11/26 13:52
@desc: 
'''

import numpy as np
import cv2
import threading
import torch
from PIL import Image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

from yolo import YOLO

def load_model():
    """
    加载训练模型
    :return:
    """
    yolo = YOLO()
    return yolo

def get_images(img_dir):
    img_names = os.listdir(img_dir)
    if len(img_names) > 200:
        img_names = img_names[:200]
    images = []
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        images.append(image)
    return img_names, images

def inference_img(model, image):
    image = Image.fromarray(np.uint8(image))
    image = np.array(model.detect_image(image))
    return image

def inference_video(video_path, model, save_pre):
    new_video_path = save_pre+video_path.split("/")[-1]
    videoCapture = cv2.VideoCapture(video_path)
    success, frame = videoCapture.read()
    i = 0
    width = videoCapture.get(3)  # float
    height = videoCapture.get(4)  # float
    new_video = cv2.VideoWriter(new_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 12, (int(width), int(height)))
    while success:
        if i % 4 == 0:
            plot_img = inference_img(model, frame)
            new_video.write(plot_img)
        success, frame = videoCapture.read()  # 获取下一帧
        i = i + 1
        if i % 480 == 0:
            print(video_path, i)
    videoCapture.release()
    new_video.release()

def inference_dir():
    model = load_model()
    img_dir = "temp/source"
    #/home/hadoop/pub_datasets/object_det/BDD100K/bdd100k_images/bdd100k/images/100k/val,temp/source
    img_names, images = get_images(img_dir)
    for img_name, image in zip(img_names, images):
        image = inference_img(model, image)
        cv2.imwrite("temp/target/pre_" + img_name, image)
        # print(img_name)

class myThread(threading.Thread):
    def __init__(self, video_path, model, save_pre):
        threading.Thread.__init__(self)
        self.video_path = video_path
        self.model = model
        self.save_pre = save_pre
    def run(self):
        print ("开始线程：" + self.video_path)
        inference_video(self.video_path, self.model, self.save_pre)
        print ("退出线程：" + self.video_path)

def inference_videos():
    model = load_model()
    video_dir = "img/vs"
    save_pre = "img/vs/result/"
    video_names = os.listdir(video_dir)
    for index, video_name in enumerate(video_names):
        video_path = video_dir + "/" + video_name
        if os.path.isdir(video_path):
            continue
        try:
            th = myThread(video_path, model, save_pre)
            th.start()
        except:
            print("处理出错", video_path)


def transfer_model():
    yolo = load_model()
    model = yolo.net
    torch.save(model.state_dict(), 'logs/Epoch76.pth', _use_new_zipfile_serialization=False)

if __name__ == '__main__':
    # inference_dir()
    inference_videos()
    # transfer_model()


