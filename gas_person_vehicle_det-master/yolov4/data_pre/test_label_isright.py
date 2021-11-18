#!/usr/bin/env python
# encoding: utf-8
'''
@author: 余智君
@license: (C) Copyright 2005-2025, 四川弘和集团.
@contact: yuzhijun@cwddd.com
@software:PyCharm2019.3
@file: test_label_isright.py
@time: 2020/11/20 19:22
@desc: 测试处理标出的数据对不对
'''

import cv2
import random
import shutil
import os

select_classs = ["person", "bicycle", "car", "motorbike", "bus", "truck"] #[person, bicycle, car, motorbike, bus, truck]
new_ids = {}
for i, select_class in enumerate(select_classs):
    new_ids[i] = select_class
print(new_ids)


def mk_dir(lable_path, dir_root="/home/hadoop/yuzhijun/dataset/view/"):
    dir_name = lable_path.split("/")[-1]
    dir_path = dir_root + dir_name
    if os.path.isdir(dir_path) is True:
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)
    return dir_path

colors = [[255,140,0], [112,128,144], [220,20,60], [255,20,147], [139,0,139], [0,0,255], [0,255,255], [0,255,127], [255,255,0], [218,165,32]] * 10
def test_label(lable_path):
    save_dir_path = mk_dir(lable_path)
    f = open(lable_path)  # 返回一个文件对象
    line = f.readline()  # 调用文件的 readline()方法
    while line:  # 处理一张图片
        if random.random() < 0.10:
            line = line.replace("\n", "")
            arr = line.split(" ")
            img_path = arr[0]
            box_cls = arr[1:]
            img = cv2.imread(img_path)
            for box_cl in box_cls:
                box = box_cl.split(",")
                if len(box) >= 5:
                    x1, y1, x2, y2, id = int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[4])
                    cv2.rectangle(img, (x1, y1), (x2, y2), colors[id], 2)
                    class_name = new_ids[id]
                    cv2.putText(img, class_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[id], 1)
            img_name = img_path.split("/")[-1]
            cv2.imwrite(save_dir_path + "/" + img_name, img)
        line = f.readline()
    f.close()


if __name__ == '__main__':
    # 测试KITTI数据集
    train_annotation_paths = [
        # '/home/hadoop/yuzhijun/dataset/目标检测/COCO2017/annotations_train2017/train2017_person_vehicle.txt',
        # '/home/hadoop/yuzhijun/dataset/目标检测/VOC2012/voc2012_person_vehicle.txt',
        # '/home/hadoop/yuzhijun/dataset/目标检测/CrowdHuman/CrowdHuman_train/CrowdHuman_train.txt',
        # '/home/hadoop/yuzhijun/dataset/目标检测/CrowdHuman/CrowdHuman_val/CrowdHuman_val.txt',
        # '/home/hadoop/yuzhijun/dataset/目标检测/CrowdHuman/CrowdHuman_train/CrowdHuman_train.txt',
        '/home/hadoop/pub_datasets/gas_station/person_vehicle_det/utill_20210114.txt',
        # '/home/hadoop/pub_datasets/object_det/BDD100K/bdd100k_labels/bdd100k/labels/100k/val_person_vehicle.txt',
        # '/home/hadoop/pub_datasets/object_det/BDD100K/bdd100k_labels/bdd100k/labels/100k/train_person_vehicle.txt',
        # '/home/hadoop/pub_datasets/object_det/VisDrone2019/VisDrone2019-DET-train/VisDrone2019-DET-train/VisDrone2019-DET-train.txt',
        # '/home/hadoop/pub_datasets/object_det/VisDrone2019/VisDrone2019-DET-val/VisDrone2019-DET-val/VisDrone2019-DET-val.txt',
        # '/home/hadoop/pub_datasets/object_det/VisDrone2019/VisDrone2019-DET-test-dev/VisDrone2019-DET-test-dev.txt'
        ]
    # lable_path = "/home/hadoop/pub_datasets/gas_station/person_vehicle_det/utill_20201202.txt"
    for train_annotation_path in train_annotation_paths:
        test_label(train_annotation_path)
        print("完成", train_annotation_path)
