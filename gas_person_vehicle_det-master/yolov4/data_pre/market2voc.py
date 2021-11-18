# -*- coding:utf-8 -*-
# @Time: 2020/12/3 15:32
# @Author: luozhoujie
# @Email: luozhoujie@cwddd.com
# @File: VOC_txt.py
# torch==0.4.1
# pycharm==2020.1

import os
import xml.etree.ElementTree as ET

c_name_map = {"oil_tank_truck":"truck",
              "truck":"truck",
              "people":"person",
              "person":"person",
              "bus":"bus",
              "car":"car"}
select_classs = ["person", "bicycle", "car", "motorbike", "bus", "truck"]
new_ids = {}
for i, select_class in enumerate(select_classs):
    new_ids[select_class] = i
print(new_ids)

def renames(xml_dirs):
    for xml_dir in xml_dirs:
        file_names = os.listdir(xml_dir)
        for file_name in file_names:
            old_name = os.path.join(xml_dir, file_name)
            new_name = old_name.replace(" ", "")
            os.rename(old_name, new_name)

def parse_xml(xml_path):
    """
    :param xml_path:
    :return: [[x1, y1, x2, y2, c_name],[x1, y1, x2, y2, c_name],...]
    """
    file_content = open(xml_path, encoding="utf-8")
    tree = ET.parse(file_content)
    root = tree.getroot()
    boxs = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        c_name = obj.find('name').text
        if int(difficult) == 1:
            continue
        xmlbox = obj.find('bndbox')
        x1 = int(xmlbox.find('xmin').text)
        y1 = int(xmlbox.find('ymin').text)
        x2 = int(xmlbox.find('xmax').text)
        y2 = int(xmlbox.find('ymax').text)
        if c_name not in c_name_map.keys(): continue
        c_id = new_ids[c_name_map[c_name]]
        boxs.append([x1, y1, x2, y2, c_id])
    return boxs

def boxs2line(img_path, boxs):
    save_line = ""  # img_path x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    if len(boxs) > 0:
        save_line = img_path
    for box in boxs:
        one_box_str = ""
        for index, num in enumerate(box):
            if index < len(box) - 1:
                one_box_str = one_box_str + str(num) + ","
            else:
                one_box_str = one_box_str + str(num)
        save_line = save_line + " " + one_box_str
    return save_line

def parse_xmls(xml_dir):
    xml_names = os.listdir(xml_dir)
    xml_names = [xml_name for xml_name in xml_names if xml_name.endswith(".xml")]
    lines = []
    for xml_name in xml_names:
        xml_path = os.path.join(xml_dir, xml_name)
        boxs = parse_xml(xml_path)
        if len(boxs) <= 0: continue
        img_path = xml_path.split(".")[0]+".jpg"
        if os.path.exists(img_path) is False: continue
        line = boxs2line(img_path, boxs)
        line = line.replace("\n", "")
        lines.append(line)
        # print(line)
        # print(xml_path, boxs)
    return lines

def parse_xml_dirs(xml_dirs):
    all_lines = []
    for xml_dir in xml_dirs:            #xml_dirs：xml第一级目录路径   xml_dir: xml文件路径
        lines = parse_xmls(xml_dir)     #xml_
        all_lines = all_lines + lines
    return all_lines

def save_txt(save_file, lines):
    if os.path.exists(save_file):
        os.remove(save_file)
    f = open(save_file, "a")
    for line in lines:
        f.write("{}\n".format(line))
    f.close()

def ck_path(file_path):
    def read_lines(file_path):
        with open(file_path) as f1:
            lines = f1.readlines()
        lines = [line.replace("\n", "") for line in lines]
        return lines
    lines = read_lines(file_path)
    for line in lines:
        img_path = line.split(" ")[0]
        if os.path.exists(img_path) is False:
            print("错误的路劲", img_path)
    print("检查完毕")


if __name__ == '__main__':
    xml_dirs = [
                "/home/hadoop/pub_datasets/gas_station/person_vehicle_det/20201126/xusha",
                "/home/hadoop/pub_datasets/gas_station/person_vehicle_det/20201127/xusha",
                "/home/hadoop/pub_datasets/gas_station/person_vehicle_det/20201130/xusha",
                "/home/hadoop/pub_datasets/gas_station/person_vehicle_det/20201201/xusha",
                "/home/hadoop/pub_datasets/gas_station/person_vehicle_det/20201202/xusha",
                "/home/hadoop/pub_datasets/gas_station/person_vehicle_det/20201203/xusha",
                "/home/hadoop/pub_datasets/gas_station/person_vehicle_det/20201204/xusha",
                "/home/hadoop/pub_datasets/gas_station/person_vehicle_det/20201207/xusha",
                "/home/hadoop/pub_datasets/gas_station/person_vehicle_det/20201208/xusha",
                "/home/hadoop/pub_datasets/gas_station/person_vehicle_det/20201209/xusha",
                "/home/hadoop/pub_datasets/gas_station/person_vehicle_det/20201211/xusha"
                ]
    dir_root = "/home/hadoop/pub_datasets/gas_station/person_vehicle_det/20210114"
    v_names = os.listdir(dir_root)          #将路径下的文件进行 生成列表
    for v_name in v_names:
        v_path = os.path.join(dir_root, v_name)     #”xusha/1“
        new_v_path = v_path.replace(" ", "")
        os.rename(v_path, new_v_path)
        xml_dirs.append(new_v_path)
    renames(xml_dirs)

    save_file = "/home/hadoop/pub_datasets/gas_station/person_vehicle_det/utill_20210114.txt"
    all_lines = parse_xml_dirs(xml_dirs)        #传入xml文件路径
    print(len(all_lines))
    save_txt(save_file, lines=all_lines)

    ck_path(file_path=save_file)
