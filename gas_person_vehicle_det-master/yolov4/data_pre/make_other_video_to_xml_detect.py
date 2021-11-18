# -*- coding:utf-8 -*-
# @Time: 2021/3/24 14:14
# @Author: luozhoujie
# @Email: luozhoujie@cwddd.com
# @File: make_other_classifier.py
# torch==0.4.1
# pycharm==2020.1

# from fire_smoke_detect.methods import FireSmokeDetection
# from pedestrian_vehicle_detection.methods import PedestrianVehicleDetection
from gas_head_det.methods import GasHeadDetect
import os
import cv2
import numpy as np

""" 用于制作其他类，并写入xml文件 """

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def write_xml(img_name, width, height, gts, xml_save_to):
    """
    用于把类别、坐标写入xml文件
    :param img_name: xml文件名，如00001
    :param width: 图像的宽
    :param height: 图像的高
    :param gts: 类别加坐标：[['cls', x1, y1, x2, y2]]，如：[['smoke', 104, 87, 264, 178], ['smoke', 311, 111, 433, 189]]
    :param xml_save_to: 保存xml文件路径
    :return:
    """
    xml_file = open((xml_save_to + '/' + str(img_name) + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>' + str(img_name) + ".jpg" + '</folder>\n')
    xml_file.write('    <filename>' + str(img_name) + '.jpg' + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')

    # write the region of image on xml file
    for gt in gts:
        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + str(gt[0]) + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(gt[1]) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(gt[2]) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(gt[3]) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(gt[4]) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')

    xml_file.write('</annotation>')
    xml_file.close()
    return


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, -1]
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1

    return keep


def py_cpu_softnms(dets, Nt=0.3, sigma=0.5, thresh=0.001, method=2):
    """
    py_cpu_softnms
    :param dets:   boexs 坐标矩阵 format [y1, x1, y2, x2]
    :param sc:     每个 boxes 对应的分数
    :param Nt:     iou 交叠门限
    :param sigma:  使用 gaussian 函数的方差
    :param thresh: 最后的分数门限
    :param method: 使用的方法
    :return:       留下的 boxes 的 index
    """

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        if i != N - 1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            # 每次把最高score的 往上拿
            dets[i, :] = dets[maxpos + i + 1, :]  # 行置换，score，area也一样
            dets[maxpos + i + 1, :] = tBD

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea

        # IoU calculate
        xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]
    # select the boxes and keep the corresponding indexes
    inds = dets[:, 4][scores > thresh]
    keep = inds.astype(int)

    return keep


def make_other(path, save_path, frames):
    """
    裁剪并扩充算法识别结果，
    @param path: 视频路径
    @param save_path: 保存图像路径
    @param frames:  控制隔多少帧取一帧，其中1秒25帧
    """
    # select_classs = ["person", "bike", "car", "motor", "bus", "truck"]
    select_classs = ["person", "head", "helmet"]
    dir_name = {0: "JPEG", 1: "XML"}                    #文件目录
    for index in range(len(dir_name)):
        if not os.path.exists(save_path + str(dir_name[index])):
            os.makedirs(save_path + str(dir_name[index]))       #如果没有 保存的路径+JPEG  或者 保存的路径+XML

    # model = PedestrianVehicleDetection(gpu=[0], batch_size=1)
    model = GasHeadDetect(gpu=[0], batch_size=1)
    for i, file in enumerate(os.listdir(path)):  #  遍历传入的视频路径 path = r"/home/hadoop/shelei_data_2/D19_renziji_1/"
        v_path = os.path.join(path, file)  # 第一次循环时，得到目录中的第一个视频
        print(v_path)
        video_load = cv2.VideoCapture(v_path)  # 加载视频
        name = file.split(".")[0]  # 取得视频名
        frame_count_id = 0  # 帧率计数器
        frame_write_id = 0  # 取帧计数器

        while True:
            ret, img = video_load.read()  # 读取视频帧
            if not ret:  # 视频结束
                break
            frame_count_id += 1         #帧率+1
            if frame_count_id % frames == 0:     #如果第25帧，则进行模型的侦测
                frame_write_id += 1
                # results = model.person_vehicle_detection([img])[0]
                results = model.detect_images([img])
                if len(results) == 0:
                    continue
                # keep = py_cpu_softnms(dets=np.array(results), method=2, thresh=0)
                # try:
                #     nms_box = np.array(results)[keep]   #得到经过nms后的框
                # except:
                #     print("error!")
                all_label = []
                all_box = []
                for i, result in enumerate(results):

                    cls = select_classs[int(result[4])]
                    if cls == "helmet":
                        continue
                    all_label.append(cls)
                    _box = [int(result[0]), int(result[1]), int(result[2]), int(result[3])]
                    all_box.append(_box)
                h, w = img.shape[:2]
                gts = [[c] + b for c, b in zip(all_label, all_box)]
                cv2.imwrite(save_path + "/JPEG/" + name + "_" + str(frame_write_id) + ".jpg", img)
                write_xml(name + "_" + str(frame_write_id), w, h, gts, save_path + "/XML/")
        print("video finished!")


if __name__ == '__main__':
    path = r"/home/hadoop/shelei_data_2/person_vehicle_dataset/xieyou/up/ch23/"
    save_path = path + "make_other/"
    print(save_path)
    # save_path = r"/home/hadoop/shelei_data_2/D19_renziji_2/make_other/"
    frames = 50
    # crop_image(path, save_path, frames)
    make_other(path, save_path, frames)











