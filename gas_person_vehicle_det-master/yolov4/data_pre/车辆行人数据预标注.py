#!bai/usr/bin/env python
# encoding: utf-8
'''
@author: 余智君
@license: (C) Copyright 2020-2030, 四川弘和集团.
@contact: yuzhijun@cwddd.com
@file: test.py
@time: 2020/12/24 17:24
@desc: 
'''


import cv2 as cv2
import os
import random
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from pedestrian_vehicle_detection.methods import PedestrianVehicleDetection

select_classs = ["person", "bike", "car", "motor", "bus", "truck"]
new_ids = {}
for i, select_class in enumerate(select_classs):
    new_ids[i] = select_class
print(new_ids)

detect = PedestrianVehicleDetection(gpu=[0], batch_size=1)

def save_txt(save_file, lines):
    # if os.path.exists(save_file):
    #     os.remove(save_file)
    f = open(save_file, "w", encoding='utf8')
    for line in lines:
        f.write("{}\n".format(line))
    f.close()

def make_xml_content(folder, filename, path, size, objects):
    """
    :param folder: 文件夹名称
    :param filename: 文件名
    :param path: 文件路径
    :param size: (w,h)图片大小
    :param objects: [{"name":"","bndbox":[x1,y1,x2,y2]},...]
    :return:
    """
    def get_object(object):
        object_xml = "\t<object>"
        object_xml += "\n\t\t<name>"+object["name"]+"</name>"
        object_xml += "\n\t\t<pose>Unspecified</pose>"
        object_xml += "\n\t\t<truncated>0</truncated>"
        object_xml += "\n\t\t<difficult>0</difficult>"
        object_xml += "\n\t\t<bndbox>"
        object_xml += "\n\t\t\t<xmin>" + str(object["bndbox"][0]) + "</xmin>"
        object_xml += "\n\t\t\t<ymin>" + str(object["bndbox"][1]) + "</ymin>"
        object_xml += "\n\t\t\t<xmax>" + str(object["bndbox"][2]) + "</xmax>"
        object_xml += "\n\t\t\t<ymax>" + str(object["bndbox"][3]) + "</ymax>"
        object_xml += "\n\t\t</bndbox>"
        object_xml += "\n\t</object>"
        return object_xml

    lines = []
    lines.append("<annotation verified=\"yes\">")
    lines.append("\t<folder>"+folder+"</folder>")
    lines.append("\t<filename>" + filename + "</filename>")
    path = "C:\\datasets\\person_vehicle\\"+folder+"\\"+filename  #服务器模式
    lines.append("\t<path>" + path + "</path>")
    lines.append("\t<source>\n\t\t<database>Unknown</database>\n\t</source>")
    lines.append("\t<size>\n\t\t<width>"+str(size[0])+"</width>\n\t\t<height>"+str(size[1])+"</height>\n\t\t<depth>3</depth>\n\t</size>")
    lines.append("\t<segmented>0</segmented>")
    for object in objects:
        object_xml = get_object(object)
        lines.append(object_xml)
    lines.append("</annotation>")
    return lines

def get_object_boxs(boxs):
    objects = []
    for box in boxs:
        x1 = min(max(box[0], 0), 1920)
        y1 = min(max(box[1], 0), 1080)
        x2 = min(max(box[2], 0), 1920)
        y2 = min(max(box[3], 0), 1080)
        id = box[4]
        dic = {}
        dic["name"] = new_ids[int(id)]
        dic["bndbox"] = [x1, y1, x2, y2]
        objects.append(dic)
    return objects

def marked_img(img_path, boxs, size=(1920, 1080)):
    if boxs is None: return
    if len(boxs) <= 0: return
    arr = img_path.split("/")
    folder = arr[-2]
    filename = arr[-1]
    path = img_path
    objects = get_object_boxs(boxs)
    lines = make_xml_content(folder, filename, path, size, objects)
    save_file = img_path.replace(".jpg", ".xml")
    save_txt(save_file, lines)

def marked_imgs(img_dir):
    img_names = os.listdir(img_dir)
    img_names = [img_name for img_name in img_names if img_name.endswith(".jpg")]
    for index, img_name in enumerate(img_names):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        boxs = detect.person_vehicle_detection([img])
        marked_img(img_path, boxs[0], size=(1920, 1080))
        if index > 0 and index % 120 == 0:
            print(img_dir, "完成:", index, "("+str(len(img_names))+")")

def marked_dirs(img_dirs, root):
    for img_dir in img_dirs:
        img_dir = os.path.join(root, img_dir)
        print("开始处理:", img_dir)
        marked_imgs(img_dir)
        print("结束处理:", img_dir)

import threading
class myThread(threading.Thread):
    def __init__(self, img_dirs, root):
        threading.Thread.__init__(self)
        self.root = root
        self.img_dirs = img_dirs
    def run(self):
        marked_dirs(self.img_dirs, self.root)


def main():
    root = "/home/hadoop/yuzhijun/dataset/GasStationData/Unloading_oil/纯卸油流程-夜间"
    all_img_dirs = os.listdir(root)
    all_img_dirs = [img_dir for img_dir in all_img_dirs if os.path.isdir(os.path.join(root, img_dir))]

    leng = len(all_img_dirs) // 4 + 1
    img_dirs0 = all_img_dirs[0: leng]
    img_dirs1 = all_img_dirs[leng: leng*2]
    img_dirs2 = all_img_dirs[leng*2: leng*3]
    img_dirs3 = all_img_dirs[leng*3: ]

    for img_dirs in [img_dirs0, img_dirs1, img_dirs2, img_dirs3]:
        th = myThread(img_dirs, root)
        th.start()


if __name__ == '__main__':
    main()








