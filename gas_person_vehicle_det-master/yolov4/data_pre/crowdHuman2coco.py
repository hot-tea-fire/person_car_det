#!/usr/bin/env python
# encoding: utf-8
'''
@author: 余智君
@license: (C) Copyright 2005-2025, 四川弘和集团.
@contact: yuzhijun@cwddd.com
@software:PyCharm2019.3
@file: crowdHuman2coco.py
@time: 2020/11/18 16:37
@desc: CrowdHuman数据集转换为COCO2017格式
输出格式
image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
'''

import json
import cv2
import os
import random

def cv2_draw_rect(img_root, img_name, gtboxe_vboxs):
    """
    可视化结果
    :param img_root:
    :param img_name:
    :param gtboxe_vboxs:
    :return:
    """
    img = cv2.imread((img_root + img_name))
    for gtboxe_vbox in gtboxe_vboxs:
        x1,y1,x2,y2 = int(gtboxe_vbox[0]), int(gtboxe_vbox[1]), int(gtboxe_vbox[2]), int(gtboxe_vbox[3])
        cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)
    img_name = img_name.split("/")[-1]
    cv2.imwrite(img_root+"/view/"+img_name, img)

def cv2_draw_rect2(img_root, save_file):
    f = open(save_file)  # 返回一个文件对象
    line = f.readline()
    while line:  # 处理一张图片
        if random.random() < 0.9:
            continue
        cols = line.split(" ")
        img_path = cols[0]
        img = cv2.imread(img_path)
        for box_str in cols[1:]:
            box_str = box_str.replace("\n", "").replace(" ", "")
            box = box_str.split(",")
            if len(box) < 4: continue
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img_name = img_path.split("/")[-1]
        save_path = img_root+"view/"+img_name
        print(save_path)
        cv2.imwrite(save_path, img)
        line = f.readline()
    f.close()


def get_CrowdHuman(img_root, file_path, set_name="train"):
    """
    解析CrowdHuman数据结构
    :param img_root:
    :param file_path:
    :param set_name:
    :return:
    """
    f = open(file_path)  # 返回一个文件对象
    line = f.readline()  # 调用文件的 readline()方法
    boxs = []
    while line: #处理一张图片
        line_dict = json.loads(line)
        img_name = "CrowdHuman_"+set_name+"/Images/"+line_dict["ID"]+".jpg"
        box = {}
        box["img_name"] = img_root + img_name
        gtboxes = line_dict["gtboxes"]
        gtboxe_tags, gtboxe_vboxs = [], []
        for gtboxe in gtboxes:  #循环处理一张图中的多个行人
            gtboxe_tag = gtboxe["tag"]
            gtboxe_vbox = gtboxe["vbox"] #x,y,w,h
            x, y, w, h = int(gtboxe_vbox[0]), int(gtboxe_vbox[1]), int(gtboxe_vbox[2]), int(gtboxe_vbox[3])
            new_gtboxe_vbox = [x, y, x+w, y+h]  #x1,y1,x2,y2
            if gtboxe_tag == "person":
                gtboxe_tags.append(gtboxe_tag)
                gtboxe_vboxs.append(new_gtboxe_vbox)
        box["gtboxe_vboxs"] = gtboxe_vboxs
        boxs.append(box)
        # print(img_name, gtboxe_vboxs),  # 后面跟 ',' 将忽略换行符
        # cv2_draw_rect(img_root, img_name, gtboxe_vboxs) #可视化结果
        line = f.readline()
    f.close()
    return boxs

def save_txt(boxs, save_file):
    """
    保存文件
    :param boxs:
    :param save_file:
    :return:
    """
    if os.path.exists(save_file):
        os.remove(save_file)
    f = open(save_file, "a")
    for box in boxs:
        img_name = box["img_name"]
        gtboxe_vboxs = box["gtboxe_vboxs"]
        gtboxe_vboxs_str = ""
        for vbox in gtboxe_vboxs:
            x1, y1, x2, y2, id = vbox[0], vbox[1], vbox[2], vbox[3], "0"
            gtboxe_vboxs_str = gtboxe_vboxs_str + str(x1)+","+str(y1)+","+str(x2)+","+str(y2)+","+id+" "
        f.write("{}\n".format(img_name+" "+gtboxe_vboxs_str))
    f.close()

if __name__ == '__main__':
    img_root = "/home/hadoop/sshfs/yuzhijun/CrowdHuman/"
    # 处理训练集
    train_file_path = "/home/hadoop/sshfs/yuzhijun/CrowdHuman/annotation_train.odgt"
    boxs = get_CrowdHuman(img_root, train_file_path, set_name="train")
    save_file = "/home/hadoop/sshfs/yuzhijun/CrowdHuman/CrowdHuman_train/CrowdHuman_train_178.txt"
    save_txt(boxs, save_file)
    # 处理验证集
    val_file_path = "/home/hadoop/sshfs/yuzhijun/CrowdHuman/annotation_val.odgt"
    boxs = get_CrowdHuman(img_root, val_file_path, set_name="val")
    save_file = "/home/hadoop/sshfs/yuzhijun/CrowdHuman/CrowdHuman_val/CrowdHuman_val_178.txt"
    save_txt(boxs, save_file)




