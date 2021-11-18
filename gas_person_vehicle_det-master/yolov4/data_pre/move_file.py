# -*- coding:utf-8 -*-
# @Time: 2020/12/24 20:07
# @Author: luozhoujie
# @Email: luozhoujie@cwddd.com
# @File: move_file.py
# torch==0.4.1
# pycharm==2020.1

import os
import shutil
import re


def move_files(path, new_path, txt):
    for line in txt:
        old = os.path.join(path, line + ".xml")
        new = os.path.join(new_path, line + ".xml")
        shutil.move(old, new)


def pipei(path, old_path, new_path, xml=".xml", jpg=".jpg"):
    for file in os.listdir(path):
        name = file.split(".")[0]
        old_paths = os.path.join(old_path, name + jpg)
        new_paths = os.path.join(new_path, name + jpg)
        shutil.move(old_paths, new_paths)
    return

#复制图片和xml到新的目录下
def move_xml(txt, old, new, name=""):
    for line in txt.readlines():
        file_name = line.strip()
        old_jpg = os.path.join(old + "/JPEG/", file_name + ".jpg")
        new_jpg = os.path.join(new + "/JPEG/", name + file_name + ".jpg")
        shutil.copy(old_jpg, new_jpg)

        old_xml = os.path.join(old + "/XML/", file_name + ".xml")
        new_xml = os.path.join(new + "/XML/", name + file_name + ".xml")
        shutil.copy(old_xml, new_xml)
    return

#取得图片名字
def find_file(path, txt):
    """ 查找文件名中的关键字，并存入文本文件 """
    for i, file in enumerate(os.listdir(path)):
        # if re.search(key, file, re.M | re.I):
        line = file[:-4]
        txt.write(line + "\n")

#更换xml路径
def move_txt(path, new, txt):
    for line in txt.readlines():
        try:
            old = os.path.join(path, line.strip() + ".xml")
            news = os.path.join(new, line.strip() + ".xml")
            shutil.move(old, news)
        except:
            print("break")

#将图片名字有空格的消除掉
def dete_space():
    path = "/package/luozhoujie/dataset/test_forest/JPEG/"
    for line in os.listdir(path):
        old = os.path.join(path, line)
        new = os.path.join(path, line.replace(" ", ""))
        shutil.move(old, new)

#将图片名字和xml名字相同的移动到新的文件下
def move_jpg_xml(orgin_jpg, orgin_xml, new_jpg, new_xml):
    """ 用于把原始路径下的jpg和xml的文件，重命名后移动到新的目录下 """
    # search = "internet"
    if not os.path.exists(new_jpg):
        os.makedirs(new_jpg)
    if not os.path.exists(new_xml):
        os.makedirs(new_xml)
    for i, files in enumerate(os.listdir(orgin_jpg)):   #对原始图片列表进行取值  假设是1.jpg

        if files.endswith(".jpg"):
            orgin_jpg_path = os.path.join(orgin_jpg,files)              #得到原始图片的路径  file:1.jpg  1.xml
            new_jpg_path = os.path.join(new_jpg,files)           #新图片路径


            orgin_xml_path = os.path.join(orgin_xml,files.replace(".jpg", ".xml"))  #老图片的xml
            new_xml_path = os.path.join(new_xml,files.replace(".jpg", ".xml")) #新图片的xml
            if os.path.isfile(orgin_xml_path):
                shutil.move(orgin_xml_path, new_xml_path)
                shutil.move(orgin_jpg_path, new_jpg_path)  # 老图片移动
    return

#将图片名字和xml名字相同的复制到新的文件下
def copy_jpg_xml(orgin_jpg, orgin_xml, new_jpg, new_xml):
    """ 用于把原始路径下的jpg和xml的文件，重命名后移动到新的目录下 """
    # search = "internet"
    if not os.path.exists(new_jpg):
        os.makedirs(new_jpg)
    if not os.path.exists(new_xml):
        os.makedirs(new_xml)
    for i, files in enumerate(os.listdir(orgin_jpg)):   #对原始图片列表进行取值  假设是1.jpg

        if files.endswith(".jpg"):
            orgin_jpg_path = os.path.join(orgin_jpg,files)              #得到原始图片的路径  file:1.jpg  1.xml
            new_jpg_path = os.path.join(new_jpg,files)           #新图片路径


            orgin_xml_path = os.path.join(orgin_xml,files.replace(".jpg", ".xml"))  #老图片的xml
            new_xml_path = os.path.join(new_xml,files.replace(".jpg", ".xml")) #新图片的xml
            if os.path.isfile(orgin_xml_path):
                shutil.copy(orgin_xml_path, new_xml_path)
                shutil.copy(orgin_jpg_path, new_jpg_path)  # 老图片移动
    return

#将XML移动到新的路径下
# def move_xml(origin_xml, new_xml, name):
#     for i, files in enumerate(os.listdir(origin_xml)):
#         if files.endswith(".xml"):
#             origin_xml_path = origin_xml + files
#             new_xml_path = new_xml + files
#             shutil.move(origin_xml_path, new_xml_path)



if __name__ == '__main__':

    origin_jpg = r"/mnt/tanmeng/shelei/data/head_detect_self_biao/head_company_imitation/jpg"
    orgin_xml = r"/mnt/tanmeng/shelei/data/head_detect_self_biao/head_company_imitation/xml"
    new_jpg = r"/mnt/tanmeng/shelei/data/head_detect_self_biao/head_company_imitation/jpg_new"
    new_xml = r"/mnt/tanmeng/shelei/data/head_detect_self_biao/head_company_imitation/xml_new"
    # name = ""
    copy_jpg_xml(origin_jpg, orgin_xml, new_jpg, new_xml)
    # origin_xml = r"E:\data\safe_hat\合并9和10"
    # new_xml = r""
    # move_xml(origin_xml, new_xml)
    # txt_file = "dete.txt"
    # f = open(txt_file, "r", encoding="utf-8").read().strip().split()
    # old = r"/package/luozhoujie/dataset/fangzhi/"
    # new = "/package/luozhoujie/dataset/dete/"
    # path = r"/package/luozhoujie/dataset/AllTrainFireSmokeDataset/XML/"

    # find_file(path, f)
    # move_txt(path, new, f)
    # dete_space()
    # if not os.path.exists(new):
    #     os.makedirs(new)
    # move_files(path, new, f)



