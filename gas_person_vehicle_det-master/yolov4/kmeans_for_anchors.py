import numpy as np
import xml.etree.ElementTree as ET
import glob
import random
import os
import cv2

def cas_iou(box,cluster):
    x = np.minimum(cluster[:,0],box[0])
    y = np.minimum(cluster[:,1],box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:,0] * cluster[:,1]
    iou = intersection / (area1 + area2 -intersection)

    return iou

def avg_iou(box,cluster):
    return np.mean([np.max(cas_iou(box[i],cluster)) for i in range(box.shape[0])])

def kmeans(box, k):
    # 取出一共有多少框
    row = box.shape[0]
    
    # 每个框各个点的位置
    distance = np.empty((row,k))
    
    # 最后的聚类位置
    last_clu = np.zeros((row,))

    np.random.seed()

    # 随机选5个当聚类中心
    cluster = box[np.random.choice(row,k,replace = False)]
    # cluster = random.sample(row, k)
    while True:
        # 计算每一行距离五个点的iou情况。
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i],cluster)
        
        # 取出最小点
        near = np.argmin(distance,axis=1)

        if (last_clu == near).all():
            break
        
        # 求每一个类的中位点
        for j in range(k):
            cluster[j] = np.median(
                box[near == j],axis=0)

        last_clu = near

    return cluster

def load_data_xml(path):
    data = []
    # 对于每一个xml都寻找box
    for xml_file in glob.glob('{}/*xml'.format(path)):
        tree = ET.parse(xml_file)
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        # 对于每一个目标都获得它的宽高
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # 得到宽高
            data.append([xmax-xmin,ymax-ymin])
    return np.array(data)

def load_data_txt(path):
    """
    余智君添加
    :param path:
    :return:
    """
    train_annotation_paths = [
        '/home/hadoop/yuzhijun/dataset/目标检测/COCO2017/annotations_train2017/train2017_person_vehicle.txt',  # COCO数据集
        '/home/hadoop/yuzhijun/dataset/目标检测/VOC2012/voc2012_person_vehicle.txt',  # voc2012
        '/home/hadoop/yuzhijun/dataset/目标检测/CrowdHuman/CrowdHuman_val/CrowdHuman_val.txt',
        '/home/hadoop/yuzhijun/dataset/目标检测/CrowdHuman/CrowdHuman_train/CrowdHuman_train.txt',
        '/home/hadoop/pub_datasets/gas_station/person_vehicle_det/utill_20201211.txt',  # 加油站数据集
        '/home/hadoop/pub_datasets/object_det/BDD100K/bdd100k_labels/bdd100k/labels/100k/val_person_vehicle.txt',
        '/home/hadoop/pub_datasets/object_det/BDD100K/bdd100k_labels/bdd100k/labels/100k/train_person_vehicle.txt',
        '/home/hadoop/pub_datasets/object_det/VisDrone2019/VisDrone2019-DET-train/VisDrone2019-DET-train/VisDrone2019-DET-train.txt',
        '/home/hadoop/pub_datasets/object_det/VisDrone2019/VisDrone2019-DET-val/VisDrone2019-DET-val/VisDrone2019-DET-val.txt',
        '/home/hadoop/pub_datasets/object_det/VisDrone2019/VisDrone2019-DET-test-dev/VisDrone2019-DET-test-dev.txt'
        ]
    train_lines = []
    for train_annotation_path in train_annotation_paths:
        if os.path.exists(train_annotation_path) is False:
            print(train_annotation_path, "不存在")
            exit(0)
        with open(train_annotation_path) as f:
            train_line = f.readlines()
            print(len(train_line), train_annotation_path)
            train_lines = train_lines + train_line
    # train_lines = train_lines[0:200]
    data = []
    for index, line in enumerate(train_lines):
        # if random.random() < (1-0.5): continue  #随机抽样50%
        #line="ch16_202009031951201731.jpg 851,8,1915,1080,5 766,212,837,409,0"
        boxs = line.split(" ")[1:]
        img_path = line.split(" ")[0]
        img = cv2.imread(img_path)
        # print(img.shape) (1080, 1920, 3)
        width, height = img.shape[1], img.shape[0]
        # 对于每一个目标都获得它的宽高
        for box_str in boxs:
            box = box_str.split(",")
            if len(box) < 4: continue
            xmin = float(box[0]) / width
            ymin = float(box[1]) / height
            xmax = float(box[2]) / width
            ymax = float(box[3]) / height
            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # 得到宽高
            data.append([xmax - xmin, ymax - ymin])
        if index % 1000 == 0 and index > 0:
            print(index, len(train_lines))
    return np.array(data)

def load_data(path):
    # data = load_data_xml(path)
    data = load_data_txt(path)
    return data

if __name__ == '__main__':
    # 运行该程序会计算'./VOCdevkit/VOC2007/Annotations'的xml
    # 会生成yolo_anchors.txt
    SIZE = 608
    anchors_num = 9
    # 载入数据集，可以使用VOC的xml
    path = r'./VOCdevkit/VOC2007/Annotations'
    
    # 载入所有的xml
    # 存储格式为转化为比例后的width,height
    data = load_data(path)
    
    # 使用k聚类算法
    out = kmeans(data, anchors_num)
    out = out[np.argsort(out[:,0])]
    print('acc:{:.2f}%'.format(avg_iou(data,out) * 100))
    print(out*SIZE)
    data = out*SIZE
    f = open("kmeans_yolo_anchors.txt", 'w')
    row = np.shape(data)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (data[i][0], data[i][1])
        else:
            x_y = ", %d,%d" % (data[i][0], data[i][1])
        f.write(x_y)
    f.close()





