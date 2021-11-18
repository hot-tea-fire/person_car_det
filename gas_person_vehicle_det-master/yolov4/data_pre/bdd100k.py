#!/usr/bin/env python
# encoding: utf-8
'''
@author: 余智君
@license: (C) Copyright 2005-2025, 四川弘和集团.
@contact: yuzhijun@cwddd.com
@software:PyCharm2019.3
@file: bdd100k.py
@time: 2020/12/2 22:28
@desc: 处理bdd100k数据集
'''
import json
import os

select_classs = ["person", "bike", "car", "motor", "bus", "truck"]
new_ids = {}
for i, select_class in enumerate(select_classs):
    new_ids[select_class] = i
print(new_ids)

def save_txt(save_file, lines):
    if os.path.exists(save_file):
        os.remove(save_file)
    f = open(save_file, "a")
    for line in lines:
        f.write("{}\n".format(line))
    f.close()

def parse_json(json_path):
    '''
      params:
        json_path -- BDD00K数据集的一个json标签文件
      return:
        返回一个列表的列表，存储了一个json文件里面的方框坐标及其所属的类，
        形如：[[325, 342, 376, 384, 'car'], [245, 333, 336, 389, 'car']]
    '''
    boxs = []
    f = open(json_path)
    info = json.load(f)
    objects = info['frames'][0]['objects']
    for i in objects:
        box = []
        if(i['category'] in select_classs):
            box.append(int(i['box2d']['x1']))
            box.append(int(i['box2d']['y1']))
            box.append(int(i['box2d']['x2']))
            box.append(int(i['box2d']['y2']))
            box.append(i['category'])
            boxs.append(box)
    return boxs

def filter_boxs(boxs):
    class_dict = {}
    for box in boxs:
        class_name = box[4]
        if class_name not in class_dict:
            class_dict[class_name] = 1
        else:
            class_dict[class_name] = class_dict[class_name] + 1
    if "truck" in class_dict.keys(): #存在卡车
        return True
    vehicle_num = 0
    for name in ["car", "motor", "bus", "truck"]:
        if name in class_dict.keys():
            vehicle_num = vehicle_num + class_dict[name]
    if "person" in class_dict.keys():
        if class_dict["person"] >= 3 and vehicle_num >= 5: #人数大于2且车辆数量大于等于5
            return True
    return False

def boxs2line(json_dir, img_name, boxs):
    #img_path=/home/hadoop/pub_datasets/object_det/BDD100K/bdd100k_images/bdd100k/images/100k
    #json_dir=/home/hadoop/pub_datasets/object_det/BDD100K/bdd100k_labels/bdd100k/labels/100k
    img_path = json_dir.replace("/bdd100k_labels/", "/bdd100k_images/").replace("/labels/", "/images/")+"/"+img_name
    save_line = ""  # img_path x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    if len(boxs) > 0:
        save_line = img_path
    for box in boxs:
        one_box_str = ""
        for index, num in enumerate(box):
            if index < len(box) - 1:
                one_box_str = one_box_str + str(num) + ","
            else:
                c_name = num
                id = new_ids[c_name]
                one_box_str = one_box_str + str(id)
        save_line = save_line + " " + one_box_str
    return save_line

def parse_jsons(json_dir):
    json_names = os.listdir(json_dir)
    lines = []
    for json_name in json_names:
        json_path = os.path.join(json_dir, json_name)
        boxs = parse_json(json_path) #[[325, 342, 376, 384, 'car'], [245, 333, 336, 389, 'car']]
        if filter_boxs(boxs) is False:
            continue
        img_name = json_name.split(".")[0]+".jpg"
        line = boxs2line(json_dir, img_name, boxs)
        # print(line)
        lines.append(line)
    print(len(lines), len(json_names))
    return lines


if __name__ == '__main__':
    json_dir = "/home/hadoop/pub_datasets/object_det/BDD100K/bdd100k_labels/bdd100k/labels/100k/val"
    lines = parse_jsons(json_dir)
    save_file = json_dir+"_person_vehicle.txt"
    save_txt(save_file, lines)

    json_dir = "/home/hadoop/pub_datasets/object_det/BDD100K/bdd100k_labels/bdd100k/labels/100k/train"
    lines = parse_jsons(json_dir)
    save_file = json_dir + "_person_vehicle.txt"
    save_txt(save_file, lines)


