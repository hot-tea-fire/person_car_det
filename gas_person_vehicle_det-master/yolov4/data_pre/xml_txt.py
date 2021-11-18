# -*- coding:utf-8 -*-
# @Time: 2020/12/3 15:32
# @Author: luozhoujie
# @Email: luozhoujie@cwddd.com
# @File: VOC_txt.py
# torch==0.4.1
# pycharm==2020.1

import os
import xml.etree.ElementTree as ET

# classes = ['fire']  # 标签
classes = ["person", "bike", "car", "motor", "bus", "truck"]
list_file = open("/home/hadoop/shelei_data_2/person_vehicle_dataset/train_data_yolov4/yinfeng/yinfeng_val_person_car.txt", "a+", encoding="utf-8")  # 写入txt文件

# 用于去除没有目标框行
# f = open("forest.txt", "a+", encoding="utf-8")
# for line in list_file.readlines():
#     line_split = line.split(" ")
#     if len(line_split) == 1:
#         print(line)
#     else:
#         f.write(line)

xml_path = r"/home/hadoop/shelei_data_2/person_vehicle_dataset/train_data_yolov4/yinfeng/image_check/"  # xml文件地址
for files in os.listdir(xml_path):
    if files.endswith(".xml"):
        xml_file = os.path.join(xml_path, files)        #将文件拼接起来
        in_file = open(xml_file, encoding="utf-8")
        tree = ET.parse(in_file)
        root = tree.getroot()
        img_file = files.replace(".xml", ".jpg")
        list_file.write(xml_path + img_file.replace("xml", "img"))   # 图像地址
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
                 int(xmlbox.find('ymax').text))
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        list_file.write('\n')
list_file.close()








