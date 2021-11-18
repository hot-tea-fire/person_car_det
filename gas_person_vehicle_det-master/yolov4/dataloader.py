from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import math
import time
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from utils.utils import bbox_iou, merge_bboxes

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from nets.yolo_training import Generator
import cv2
import os

class YoloDataset(Dataset):
    def __init__(self, train_lines, image_size, mosaic=True):
        super(YoloDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.mosaic = mosaic
        self.flag = True

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_img(self, input_shape, img_path, jitter):
        image = Image.open(img_path)
        iw, ih = image.size
        h, w = input_shape

        # 调整图片大小
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.5, 1.5)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 放置图片
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        new_image.paste(image, (dx, dy))
        image = new_image
        return image

    # def get_random_data_bak(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
    #     """实时数据增强的随机预处理"""
    #     line = annotation_line.split()
    #     img_path = line[0]
    #     image = self.get_img(input_shape, img_path, jitter)
    #
    #     # 是否翻转图片
    #     flip = self.rand() < .5
    #     if flip:
    #         image = image.transpose(Image.FLIP_LEFT_RIGHT)
    #
    #     # 色域变换
    #     hue = self.rand(-hue, hue)
    #     sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
    #     val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
    #     x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
    #     x[..., 0] += hue*360
    #     x[..., 0][x[..., 0]>1] -= 1
    #     x[..., 0][x[..., 0]<0] += 1
    #     x[..., 1] *= sat
    #     x[..., 2] *= val
    #     x[x[:,:, 0]>360, 0] = 360
    #     x[:, :, 1:][x[:, :, 1:]>1] = 1
    #     x[x<0] = 0
    #     image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
    #
    #     # 调整目标框坐标
    #     box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
    #     box_data = np.zeros((len(box), 5))
    #     if len(box) > 0:
    #         np.random.shuffle(box)
    #         box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
    #         box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
    #         if flip:
    #             box[:, [0, 2]] = w - box[:, [2, 0]]
    #         box[:, 0:2][box[:, 0:2] < 0] = 0
    #         box[:, 2][box[:, 2] > w] = w
    #         box[:, 3][box[:, 3] > h] = h
    #         box_w = box[:, 2] - box[:, 0]
    #         box_h = box[:, 3] - box[:, 1]
    #         box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框
    #         box_data = np.zeros((len(box), 5))
    #         box_data[:len(box)] = box
    #     if len(box) == 0:
    #         return image_data, []
    #
    #     if (box_data[:, :4] > 0).any():
    #         return image_data, box_data
    #     else:
    #         return image_data, []

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        """实时数据增强的随机预处理"""
        # 按照原始比例把图片缩小，增强并放入一张目标尺寸的图片内，并对真值框也进行相应调整

        line = annotation_line.split()
        try:
            image = Image.open(line[0])
        except:
            print(annotation_line)
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        # 调整图片大小
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 放置图片
        # 在目标图片的尺寸和单张图片变化后的插值之间随机产生一个偏移作为放置图片的位置
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h),
                              (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 是否翻转图片
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 色域变换
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = rgb_to_hsv(np.array(image) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image_data = hsv_to_rgb(x) * 255  # numpy array, 0 to 1

        # 调整目标框坐标
        # voc数据集 包含左下角和右上角xy坐标
        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            np.random.shuffle(box)
            # 计算出图片变化以后真值框的的位置
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box
        if len(box) == 0:
            return image_data, []

        if (box_data[:, :4] > 0).any():
            return image_data, box_data
        else:
            return image_data, []


    #增加数据的 mosaic处理
    def get_random_data_with_Mosaic(self, annotation_line, input_shape, hue=.1, sat=1.5, val=1.5):
        h, w = input_shape
        min_offset_x = 0.3
        min_offset_y = 0.3
        scale_low = 1 - min(min_offset_x, min_offset_y)
        scale_high = scale_low + 0.2

        image_datas = []
        box_datas = []
        index = 0

        place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
        place_y = [0, int(h * min_offset_y), int(h * min_offset_y), 0]
        for i, line in enumerate(annotation_line):
            # 每一行进行分割
            line_content = line.split()
            if len(line_content) == 0:
                print(i, line)

            image = Image.open(line_content[0])


            image = image.convert("RGB")
            # 图片的大小
            iw, ih = image.size
            # 保存框的位置
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

            # 是否翻转图片
            flip = self.rand() < 0.5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            # 对输入进来的图片进行缩放
            new_ar = w / h
            scale = self.rand(scale_low, scale_high)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # 进行色域变换
            hue = self.rand(-hue, hue)
            sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
            val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
            x = cv2.cvtColor(np.array(image, np.float32)/255, cv2.COLOR_RGB2HSV)
            x[..., 0] += hue*360
            x[..., 0][x[..., 0]>1] -= 1
            x[..., 0][x[..., 0]<0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x[:,:, 0]>360, 0] = 360
            x[:, :, 1:][x[:, :, 1:]>1] = 1
            x[x<0] = 0
            image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) # numpy array, 0 to 1

            image = Image.fromarray((image * 255).astype(np.uint8))
            # 将图片进行放置，分别对应四张分割图片的位置
            dx = place_x[index]
            dy = place_y[index]
            new_image = Image.new('RGB', (w, h), (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            # 对box进行重新处理
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)


        # 将图片分割，放在一起
        cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
        cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        # 对框进行进一步的处理
        new_boxes = np.array(merge_bboxes(box_datas, cutx, cuty))

        if len(new_boxes) == 0:
            return new_image, []
        if (new_boxes[:, :4] > 0).any():
            return new_image, new_boxes
        else:
            return new_image, []

    def __getitem__(self, index):
        st = time.time()
        lines = self.train_lines
        n = self.train_batches
        index = index % n
        if self.mosaic:
            if self.flag and (index + 4) < n:
                img, y = self.get_random_data_with_Mosaic(lines[index:index + 4], self.image_size[0:2])
            else:
                img, y = self.get_random_data(lines[index], self.image_size[0:2])
            self.flag = bool(1-self.flag)
        else:
            img, y = self.get_random_data(lines[index], self.image_size[0:2])

        if len(y) != 0:
            # 从坐标转换成0~1的百分比
            boxes = np.array(y[:, :4], dtype=np.float32)
            boxes[:, 0] = boxes[:, 0] / self.image_size[1]
            boxes[:, 1] = boxes[:, 1] / self.image_size[0]
            boxes[:, 2] = boxes[:, 2] / self.image_size[1]
            boxes[:, 3] = boxes[:, 3] / self.image_size[0]

            boxes = np.maximum(np.minimum(boxes, 1), 0)
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
            boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
            y = np.concatenate([boxes, y[:, -1:]], axis=-1)

        img = np.array(img, dtype=np.float32)

        tmp_inp = np.transpose(img / 255.0, (2, 0, 1))
        tmp_targets = np.array(y, dtype=np.float32)

        return tmp_inp, tmp_targets


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes



if __name__ == '__main__':
    train_annotation_paths = [
        # '/home/hadoop/yuzhijun/dataset/目标检测/COCO2017/annotations_train2017/train2017_person_vehicle.txt',  # COCO数据集
        # '/home/hadoop/yuzhijun/dataset/目标检测/VOC2012/voc2012_person_vehicle.txt',  # voc2012
        # '/home/hadoop/yuzhijun/dataset/目标检测/CrowdHuman/CrowdHuman_val/CrowdHuman_val.txt',
        # '/home/hadoop/yuzhijun/dataset/目标检测/CrowdHuman/CrowdHuman_train/CrowdHuman_train.txt',
        # '/home/hadoop/pub_datasets/gas_station/person_vehicle_det/utill_20210114.txt',  # 加油站数据集
        # '/home/hadoop/pub_datasets/object_det/BDD100K/bdd100k_labels/bdd100k/labels/100k/val_person_vehicle.txt',
        # # bdd100k
        # '/home/hadoop/pub_datasets/object_det/BDD100K/bdd100k_labels/bdd100k/labels/100k/train_person_vehicle.txt',
        # # bdd100k
        # '/home/hadoop/pub_datasets/object_det/VisDrone2019/VisDrone2019-DET-train/VisDrone2019-DET-train/VisDrone2019-DET-train.txt',
        # # VisDrone2019
        # '/home/hadoop/pub_datasets/object_det/VisDrone2019/VisDrone2019-DET-val/VisDrone2019-DET-val/VisDrone2019-DET-val.txt',
        # # VisDrone2019
        # '/home/hadoop/pub_datasets/object_det/VisDrone2019/VisDrone2019-DET-test-dev/VisDrone2019-DET-test-dev.txt',
        # # VisDrone2019

        '/home/hadoop/shelei_data_2/person_vehicle_dataset/train_data_yolov4/duyuting/duyuting_val_person_car_146.txt',
        # # 佘磊添加
        # '/home/hadoop/shelei_data_2/person_vehicle_dataset/train_data_yolov4/xusha/xusha_val_person_car_146.txt',  #无问题
        # '/home/hadoop/shelei_data_2/person_vehicle_dataset/train_data_yolov4/yinfeng/yinfeng_val_person_car_146.txt' #无问题
    ]
    # YoloDataset()
    train_lines = []
    input_shape = (608, 608)
    mosaic = True
    Batch_size = 16
    for train_annotation_path in train_annotation_paths:
        if os.path.exists(train_annotation_path) is False:
            print(train_annotation_path, "不存在")
            exit(0)
        with open(train_annotation_path) as f:  # 读取一个txt
            print(train_annotation_path)
            train_line = f.readlines()
            print(len(train_line), train_annotation_path)
            train_lines = train_lines + train_line
            # train_lines = train_lines[:2000]
    train_dataset = YoloDataset(train_lines, (input_shape[0], input_shape[1]), mosaic=mosaic)
    # gen = DataLoader(train_dataset, shuffle=False, batch_size=1,
    #                  drop_last=True, collate_fn=yolo_dataset_collate)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=16, pin_memory=True,
                     drop_last=True, collate_fn=yolo_dataset_collate)
    for i, batch in enumerate(gen):
        # b = batch
        print("i:",i)


