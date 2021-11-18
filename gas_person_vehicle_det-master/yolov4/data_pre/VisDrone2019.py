#!/usr/bin/env python
# encoding: utf-8
'''
@author: 余智君
@license: (C) Copyright 2005-2025, 四川弘和集团.
@contact: yuzhijun@cwddd.com
@software:PyCharm2019.3
@file: VisDrone2019.py
@time: 2020/12/2 11:09
@desc: VisDrone2019 数据集处理成voc格式
'''

import os

# name_dict = {'0': 'ignored regions', '1': 'pedestrian', '2': 'people',
#              '3': 'bicycle', '4': 'car', '5': 'van', '6': 'truck',
#              '7': 'tricycle', '8': 'awning-tricycle', '9': 'bus',
#              '10': 'motor', '11': 'others'}
name_dict = {'0': 'ignored regions', '1': 'person', '2': 'person',
             '3': 'bicycle', '4': 'car', '5': 'van', '6': 'truck',
             '7': 'bicycle', '8': 'bicycle', '9': 'bus',
             '10': 'motorbike', '11': 'others'}
select_classs = ["person", "bicycle", "car", "motorbike", "bus", "truck"]
new_ids = {}
for i, select_class in enumerate(select_classs):
    new_ids[select_class] = i
print(new_ids)

g_ids = {}

def read_lines(file_path):
    with open(file_path) as f1:
        lines = f1.readlines()
    lines = [line.replace("\n", "") for line in lines]
    return lines

def save_txt(save_file, lines):
    if os.path.exists(save_file):
        os.remove(save_file)
    f = open(save_file, "a")
    for line in lines:
        f.write("{}\n".format(line))
    f.close()

def load_txt(txt_path, img_path):
    """ x1,y1,w,h,class_id
    819,315,109,35,0,0,0,0
    825,340,21,31,0,0,0,0
    886,350,34,42,0,0,0,0
    :param txt_path:
    :return:
    """
    boxs = []
    lines = read_lines(txt_path)
    for line in lines:
        box_arr = line.split(",")
        box = [int(i) for i in box_arr[:4]]
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        id = box_arr[5]
        if id not in g_ids.keys(): g_ids[id] = 1
        else: g_ids[id] = g_ids[id] + 1
        # print(name_dict[id])
        if name_dict[id] in select_classs:
            new_id = new_ids[name_dict[id]]
            box.append(new_id)
            boxs.append(box)
    save_line = "" #img_path x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    if len(boxs) > 0:
        save_line = img_path
    for box in boxs:
        one_box_str = ""
        for index, num in enumerate(box):
            if index < len(box) - 1:
                one_box_str = one_box_str + str(num) + ","
            else: one_box_str = one_box_str + str(num)
        save_line = save_line + " " + one_box_str
    return save_line

def load_txts(data_dir):
    txt_names = os.listdir(data_dir+"/annotations")
    img_dir = data_dir + "/images/"
    lines = []
    for txt_name in txt_names:
        txt_path = os.path.join(data_dir+"/annotations", txt_name)
        img_path = img_dir + txt_name.replace(".txt", ".jpg")
        save_line = load_txt(txt_path, img_path)
        if save_line == "": continue
        lines.append(save_line)
    return lines

def save_txt(lines, save_file):
    """
    保存文件
    :return:
    """
    if os.path.exists(save_file):
        os.remove(save_file)
    f = open(save_file, "a")
    for line in lines:
        f.write("{}\n".format(line))
    f.close()

if __name__ == '__main__':
    data_dir = "/home/hadoop/pub_datasets/object_det/VisDrone2019/VisDrone2019-DET-val/VisDrone2019-DET-val"
    save_file = data_dir + "/" + data_dir.split("/")[-1] + ".txt"
    lines = load_txts(data_dir)
    save_txt(lines, save_file)

    data_dir = "/home/hadoop/pub_datasets/object_det/VisDrone2019/VisDrone2019-DET-train/VisDrone2019-DET-train"
    save_file = data_dir + "/" + data_dir.split("/")[-1] + ".txt"
    lines = load_txts(data_dir)
    save_txt(lines, save_file)

    data_dir = "/home/hadoop/pub_datasets/object_det/VisDrone2019/VisDrone2019-DET-test-dev"
    save_file = data_dir + "/" + data_dir.split("/")[-1] + ".txt"
    lines = load_txts(data_dir)
    save_txt(lines, save_file)

    print(g_ids)



