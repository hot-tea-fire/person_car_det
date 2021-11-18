# -*- coding: utf-8 -*-
"""
@Author  : quedoulin
@Email   : quedoulin@cwddd.com
@project : extinguisher
@File    : crop_sit_rail_from_videos.py
@Time    : 2021/4/25
@software: PyCharm
佘磊要求：截取加油站的person(非员工，最好是站立、完整的图片，不要半身截图)
    （越多越好， 先跑移动硬盘上有的，再跑服务器上的加油区）
"""
import os
import datetime
import math
import cv2
from torch.multiprocessing import Pool, set_start_method
import numpy as np
import torch

# global skip_frame
skip_frame = 200

points_mapper = {
    'asbestos_mat': {
        'ch23': [[[886, 623], [1027, 638], [999, 746], [859, 722]],  # 灭火毯
                 [[881, 593], [1100, 623], [1048, 752], [822, 723]]],
        'D19': [[[807, 671], [962, 677], [947, 767], [794, 761]],  # 灭火毯
                [[814, 647], [1014, 660], [1000, 797], [792, 782]]]
    },
    'oil_interface': {
        'ch23': [[[934, 436], [959, 390], [1022, 419], [1003, 471]],  # 柴油口
                 [[910, 484], [905, 519], [955, 527], [965, 504]],  # 95口
                 [[941, 450], [927, 477], [973, 505], [997, 486]]],  # 92口
        'D19': [[[853, 473], [909, 504], [888, 546], [830, 503]],  # 柴油口
                [[832, 535], [815, 555], [869, 581], [884, 564]],  # 95口
                [[842, 515], [831, 533], [881, 562], [894, 544]]]  # 92口
    },
    'oil_gas_interface': {
        'ch23': [[1114, 98], [1161, 102], [1153, 157], [1104, 143]],
        'D19': [[925, 284], [955, 278], [954, 323], [920, 321]]
    },
    'oil_tube': {
        # 永兴服务区A2号加油机
        'ch04': [[[773, 367], [838, 370], [779, 565], [713, 557]]],
        # 永兴服务区A1号加油机
        'ch14': [[[1133, 312], [1185, 308], [1232, 451], [1180, 462]]],
        # 永兴服务区A1号加油机
        'ch15': [[[693, 273], [755, 275], [714, 430], [640, 431]]],
        # 永兴服务区A2号加油机
        'ch17': [[[1284, 376], [1337, 379], [1379, 536], [1325, 537]]],
        # 4号加油区
        'ch20': [[[981, 324], [1060, 330], [1022, 459], [920, 458]],
                 [[1366, 343], [1447, 346], [1501, 492], [1402, 494]]],
        # 6号加油区
        'ch21': [[[653, 292], [737, 292], [720, 442], [610, 439]],
                 [[1051, 299], [1133, 298], [1216, 437], [1112, 444]]],
        # 5号加油区
        'ch22': [[[418, 48], [493, 49], [449, 201], [345, 198]],
                 [[808, 62], [897, 57], [947, 212], [844, 219]]],
        # 7号加油区
        'ch25': [[[1320, 470], [1422, 474], [1269, 604], [1156, 597]]],
        # 永兴服务区B7号加油机
        'D05': [[[1152, 255], [1211, 260], [1111, 353], [1037, 347]]],
        # 永兴服务区B云台
        'D07': [[[474, 511], [583, 504], [818, 619], [699, 622]]],
        # 永兴服务区B4号加油机
        'D09': [[[819, 400], [881, 400], [850, 538], [758, 538]],
                [[1133, 403], [1199, 400], [1251, 525], [1165, 528]]],
        # 永兴服务区B5号加油机
        'D10': [[[797, 184], [847, 187], [812, 287], [749, 290]],
                [[1044, 186], [1096, 186], [1125, 282], [1057, 285]]],
        # 永兴服务区B1号加油机
        'D12': [[[1191, 280], [1256, 280], [1305, 444], [1231, 446]]],
        # 永兴服务区B2号加油机
        'D14': [[[1172, 395], [1223, 396], [1248, 587], [1190, 584]]],
        # 永兴服务区B1号 加油机
        'D16': [[[698, 459], [775, 450], [759, 621], [682, 633]]],
        # 下行4号加油机
        'D20': [[[744, 342], [821, 323], [768, 468], [664, 466]],
                [[1044, 186], [1096, 186], [1125, 282], [1057, 285]]],
        # 下行6号加油机
        'D21': [[[755, 356], [820, 358], [777, 493], [677, 492]],
                [[1119, 362], [1197, 360], [1246, 493], [1150, 491]]],
        # 下行5号加油机
        'D22': [[[713, 333], [795, 333], [769, 480], [653, 480]],
                [[1114, 337], [1193, 340], [1260, 474], [1149, 488]]],
        # 下行7号加油机
        'D25': [[[1326, 382], [1418, 390], [1259, 508], [1145, 500]]]
    },
    'sit_rail': {
        # 永兴服务区A2号加油机
        'ch04': [[[302, 595], [691, 595], [693, 792], [301, 790]]],
        'ch05': [[[748, 788], [1142, 789], [1142, 988], [747, 989]]],
        'ch07': [[433, 269], [635, 269], [433, 435], [635, 435]],
        'ch10': [[[653, 538], [762, 539], [761, 661], [653, 660]]],
        'ch12': [[[1225, 522], [1416, 523], [1417, 706], [1225, 705]]],
        # renziji add ch13
        'ch13': [[[631, 290], [1037, 300], [1050, 512], [644, 496]]],
        # 永兴服务区A1号加油机
        'ch14': [[[1193, 342], [1518, 344], [1517, 506], [1193, 505]]],
        # 永兴服务区A1号加油机
        'ch15': [[[353, 314], [681, 315], [684, 480], [353, 483]]],
        # 永兴服务区A2号加油机
        'ch17': [[[1383, 600], [1746, 600], [1748, 773], [1384, 776]]],
        # 4号加油区
        'ch20': [[[995, 128], [1433, 127], [1430, 353], [994, 353]]],
        # 6号加油区
        'ch21': [[[651, 78], [1136, 79], [1138, 312], [651, 310]]],
        # 5号加油区
        'ch22': [#[[418, 48], [493, 49], [449, 201], [345, 198]],
                 [[808, 62], [897, 57], [947, 212], [844, 219]]],
        # 7号加油区
        'ch25': [[[1219, 237], [1426, 238], [1430, 474], [1218, 468]]],
        'D03': [[[853, 366], [1233, 363], [1240, 543], [849, 534]]],
        'D04': [[[722, 282], [864, 282], [870, 439], [723, 437]]],
        # 永兴服务区B7号加油机
        'D05': [[[1096, 111], [1229, 108], [1231, 249], [1097, 249]]],
        # 永兴服务区B云台
        'D07': [[[405, 213], [663, 211], [664, 498], [405, 499]]],
        # 永兴服务区B4号加油机
        'D09': [[[823, 226], [1194, 229], [1193, 401], [824, 402]]],
        # 永兴服务区B5号加油机
        'D10': [[[796, 54], [1085, 55], [1089, 185], [797, 187]]],
        # 永兴服务区B1号加油机
        'D12': [[[1232, 308], [1602, 311], [1605, 483], [1237, 487]]],
        # 永兴服务区B2号加油机
        'D14': [[[1186, 252], [1296, 268], [1270, 408], [1160, 377]]],
        # 永兴服务区B1号 加油机
        'D16': [[[322, 548], [733, 478], [756, 684], [349, 750]]],
        # 下行4号加油机
        'D20': [[[744, 127], [1165, 129], [1167, 348], [744, 345]]],
        # 下行6号加油机
        'D21': [[[761, 171], [1190, 174], [1189, 369], [760, 365]]],
        # 下行5号加油机
        'D22': [[[740, 131], [1173, 131], [1173, 338], [741, 338]]],
        # 下行7号加油机
        'D25': [[[1235, 124], [1444, 125], [1447, 373], [1236, 372]]]
    }
}

model_class_mapper = {'asbestos_mat': {0: 'No_Asbestos', 1: 'No_Extinguisher', 2: 'Asbestos', 3: 'Extinguisher'},
                      'person_car': {0: "person", 1: "bicycle", 2: "car", 3: "motorbike", 4: "bus", 5: "truck"},
                      'oil_gas_interface': {0: 'no', 1: 'other', 2: 'yes'},
                      'oil_interface': {0: 'No', 1: 'Other', 2: 'Slate', 3: 'Yes'},
                      'oil_tube': {0: 'no', 1: 'no_cheliang', 2: 'no_jiayou', 3: 'no_other', 4: 'no_yingzi', 5: 'yes'},
                      'sit_rail': {0: 'Person', 1: 'Sit_Person', 2: 'Other'},
                      'staff_cloths': {0: 'Blue', 1: 'Dark', 2: 'Red', 3: 'Yellow', 4: 'Other', 5: 'Object', 6: 'White', 7: 'Half'}
                      }
# 'staff_cloths':
# 0代表浅蓝色；1代表深蓝色+灰色服装  2代表红色服装 ，3代表黄色服装，4代表非员工（普通行人），5代表非人（行人误检）
#                       6代表白色员工服，7代表人只有头部，下肢或者分类不明确

def model_deal(model_name, img):
    if model_name == 'asbestos_mat':
        # from extinguisher_mat_classification.methods import ExtinguisherMatClassification
        # detector = ExtinguisherMatClassification(gpu=[0])
        # output = detector.extinguisher_mat_classification([img])[0]
        # # from my_test import ExtinguisherMatClassification
        # # detector = ExtinguisherMatClassification(gpus=[0])
        # # output = detector.extinguisher_mat_classification([img])[0]
        # return output
        return None
    elif model_name == 'oil_interface':
        # from oil_interface_detection.methods import OilInterfaceClassifier
        # detector = OilInterfaceClassifier(gpus=[0])
        # output = detector.oil_interface_detect([img])
        # return output
        return None
    elif model_name == 'oil_gas_interface':
        # from oil_gas_interface_detection.methods import OilGasInterfaceClassifier
        # detector = OilGasInterfaceClassifier(gpus=[0])
        # output = detector.oil_gas_interface_detect([img])
        # return output
        return None
    elif model_name == 'oil_tube':
        # from oil_tube_abnormal_classify.methods import OilTubeAbnormalClassifier
        # detector = OilTubeAbnormalClassifier(gpu=[0])
        # output = detector.oil_tube_abnormal_classify([img])[0]
        # return output
        return None
    elif model_name == 'person_car':
        from pedestrian_vehicle_detection.methods import PedestrianVehicleDetection
        detector = PedestrianVehicleDetection(gpu=[4])
        output = detector.person_vehicle_detection([img])[0]
        return output
    elif model_name == 'sit_rail':
        # from sit_rail_classify.methods import SitRailClassifier
        # detector = SitRailClassifier(gpus=[0])
        # output = detector.sit_rail_classify([img])[0]
        # return output
        return None
    elif model_name == 'staff_cloths':
        from staff_classification.methods import StaffClassification
        detector = StaffClassification(gpu=[4])
        output = detector.detect([img])[0]
        return output


def min_distance_rectamgles(box1, box2):
    box1_w = box1[2] - box1[0]
    box1_h = box1[2] - box1[0]
    box2_w = box2[2] - box2[0]
    box2_h = box2[3] - box2[1]
    center1_x = box1[0] + ((box1[2] - box1[0]) / 2)
    center1_y = box1[1] + ((box1[3] - box1[1]) / 2)
    center2_x = box2[0] + ((box2[2] - box2[0]) / 2)
    center2_y = box2[1] + ((box2[3] - box2[1]) / 2)

    dx = abs(center2_x - center1_x)
    dy = abs(center2_y - center1_y)

    # 两矩形不相交，在X轴方向有部分重合的两个矩形，最小距离是上矩形的下边线与下矩形的上边线之间的距离
    if (dx < ((box1_w + box2_w) / 2)) and (dy >= ((box1_h + box2_h) / 2)):
        min_dist = dy - ((box1_h + box2_h) / 2)
    # 两矩形不相交，在Y轴方向有部分重合的两个矩形，最小距离是左矩形的右边线与右矩形的左边线之间的距离
    elif (dx > ((box1_w + box2_w) / 2)) and (dy < ((box1_h + box2_h) / 2)):
        min_dist = dx - ((box1_w + box2_w) / 2)

    elif (dx >= ((box1_w + box2_w) / 2)) and (dy >= ((box1_h + box2_h) / 2)):
        delta_x = dx - ((box1_w + box2_w) / 2)
        delta_y = dy - ((box1_h + box2_h) / 2)
        min_dist = math.sqrt(delta_x * delta_x + delta_y * delta_y)
    else:
        min_dist = -1

    return min_dist


def ImageCut(img, points):
    mat_x, mat_y, mat_w, mat_h = cv2.boundingRect(np.array(points, np.int32))
    mat = img[mat_y:mat_y + mat_h, mat_x:mat_x + mat_w]
    mat_mask = np.zeros(mat.shape, np.uint8)
    mat_point = [(i - mat_x, j - mat_y) for (i, j) in points]
    mat_mask1 = cv2.polylines(mat_mask, [np.array(mat_point)], True, (0, 255, 255))  # 画多边形
    mat_mask2 = cv2.fillPoly(mat_mask1, [np.array(mat_point)], (255, 255, 255))  # 填充多边形
    crop_img = cv2.bitwise_and(mat_mask2, mat)
    return crop_img


def DrawBoxes(img, p, content):
    mat_x, mat_y = p
    font = cv2.FONT_HERSHEY_SIMPLEX
    #           照片  添加的文字    /左上角坐标   字体           字体大小        颜色        字体粗细
    cv2.putText(img, content, org=(mat_x, mat_y + 30), fontFace=font, fontScale=1, color=(0, 255, 0))
    # cv2.rectangle(img, (mat_x, mat_y), (mat_x + mat_w, mat_y + mat_h), (0, 255, 0), 3)


def ImageReatCut(img, points):
    crop_img = img[points[1]:points[3], points[0]:points[2]]
    return crop_img

#   VideHandle('sit_rail', camera_id, path, out)
def VideoHandle(model_name: str, camera_id: str, video_path: str, output: str):
    # points_mapper[model_name][camera_id]
    # if camera_id not in points_mapper[model_name]:
    #     print(2*"\n")
    #     print(f"{camera_id} not in points_mapper['{model_name}']" )
    #     print(2*"\n")
    #     return
    filename, video_format = os.path.basename(video_path).split('.')
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f'读取失败，请检查视频{video_path}')
        with open("./err.log", 'wb') as f:
            f.write(video_path)
    total_frame = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    current_frame = 0
    person_count = 0
    while video_capture.isOpened():
        ret, img = video_capture.read()
        if not ret:
            break
        current_frame += 1
        global skip_frame
        if current_frame % skip_frame == 0:
            # core part
            # det is like this:
            # [[813, 409, 844, 488, 0], [40, 398, 169, 453, 2], [895, 294, 1624, 497, 5], [129, 339, 239, 428, 5], [738, 313, 905, 453, 5]]
            det = model_deal('person_car', img)
            if det is None:
                continue
            for i, result in enumerate(det):
                class_name = model_class_mapper['person_car'][result[-1]]
                if class_name != 'person':
                    continue
                box = result[:4]
                box[0] = max(0, box[0])
                box[1] = max(0, box[1])
                box[2] = max(0, box[2])
                box[3] = max(0, box[3])
                # 从一个frame中截出一个人
                crop_img = ImageReatCut(img, box)
                # cv2.imshow(class_name, crop_img)
                # cv2.waitKey(0)
                # 判断crop_img是否是员工，这里只要非员工
                cloth_result = model_deal('staff_cloths', crop_img)
                predict, value = cloth_result
                # if model_class_mapper['staff_cloths'][predict] not in ['Blue', 'Red', 'Yellow', 'White']:
                if model_class_mapper['staff_cloths'][predict] != 'Other':
                    continue
                t = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                image_filename = filename + '_' + str(t) + '_' + '.jpg'
                save_path = os.path.join(output, class_name)
                os.makedirs(save_path, exist_ok=True)
                cropped_image_path = os.path.join(save_path, image_filename)
                cv2.imwrite(cropped_image_path, crop_img)
                print(f"{class_name} 图片已存储在路径：{cropped_image_path}")
                person_count += 1

                # points = points_mapper[model_name][camera_id]
                # mat_x, mat_y, mat_w, mat_h = cv2.boundingRect(np.array(points, np.int32))
                # if result[4] == 0 and (
                #         min_distance_rectamgles(box, [mat_x, mat_y, mat_x + mat_w, mat_y + mat_h]) < 100):
                #     crop_img = ImageReatCut(img, box)
                #     # result = model_deal(model_name, crop_img)
                #     # predict, value = result
                #     # print(f"预测: {model_class_mapper[model_name][predict]} 置信度: {value}")
                #     # cv2.imshow('', crop_img)
                #     # cv2.waitKey(0)
                #     t = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                #     image_filename = filename + '_' + str(t) + '_' + '.jpg'
                #     save_path = os.path.join(output, model_class_mapper[model_name][predict])
                #     os.makedirs(save_path, exist_ok=True)
                #     cv2.imwrite(os.path.join(save_path, image_filename), crop_img)
                #     print(f"图片已存储在路径：{os.path.join(save_path, image_filename)}")

        if total_frame == current_frame:
            print(f'{video_path}---视频已截取完成')
            break
    video_capture.release()
    print(3*'\n')
    print(f'Have cropped {person_count} person images from {video_path}')
    print(3*'\n')


if __name__ == '__main__':
    # # 坐防撞栏，crop frames
    # root = r"/home/hadoop/PycharmProject/Data/video/sit_20210429"
    # root = r"E:\20210607_chouyan_phone_fangzhuanglan\sit_rail"
    # root = r"E:\20210607_chouyan_phone_fangzhuanglan\test_sit_rail"
    # root = r"E:\20210607_chouyan_phone_fangzhuanglan\sit_rail"
    # root = r"E:\20210607_chouyan_phone_fangzhuanglan\other_videos"

    # out = r"/home/hadoop/PycharmProject/Data/20210429_sit_rail"
    # out = r"E:\20210607_chouyan_phone_fangzhuanglan\sit_rail_output"
    # out = r"E:\20210607_chouyan_phone_fangzhuanglan\test_sit_rail_output"
    # out = r"E:\20210607_chouyan_phone_fangzhuanglan\sit_rail_output"
    # out = r"E:\20210607_chouyan_phone_fangzhuanglan\other_videos_sit_rail_output"

    # set_start_method('spawn')
    # results = []
    # pool = Pool(processes=4)
    # for camera_id in os.listdir(root):
    #     file_path = os.path.join(root, camera_id)
    #     if not os.path.isdir(file_path):
    #         continue
    #     for video_name in os.listdir(file_path):
    #         if video_name.split('.')[-1] == 'jpg':
    #             continue
    #         path = os.path.join(file_path, video_name)
    #         print(path)
    #         VideoHandle('sit_rail', camera_id, path, out)
    #         results.append(pool.apply_async(VideoHandle, args=('sit_rail', camera_id, path, out)))
    # pool.close()
    # pool.join()
    # for result in results:
    #     print('image read failed', result.get())

    # crop persons from videos
    # folders = [r"E:\20210607_chouyan_phone_fangzhuanglan\ch01wuyong"]
    # folders = [r"E:\20210607_chouyan_phone_fangzhuanglan\ch11"]

    # print(torch.cuda.current_device())

    # folders = [r"E:\20210607_chouyan_phone_fangzhuanglan\ch11wuyong"]
    # out = r"G:\renziji_dataset\persons_shelei"

    # folders = [r"/mnt/md0/AIdata/SourceData/原始数据_加油站/中路能源/永兴加油站/加油区"]
    # out = r"/mnt/md0/AIdata/SourceData/原始数据_加油站/中路能源/永兴加油站/cropped_person_images_renziji"

    # folders = [r"/home/hadoop/PycharmProject/Data/video/ch21wuyong"]
    # out = r"/home/hadoop/PycharmProject/Data/video/ch21wuyong_cropped_persons_renziji"

    # folders = [r"/home/hadoop/PycharmProject/Data/video/cigarette_videos"]
    # out = r"/home/hadoop/PycharmProject/Data/video/cigarette_videos_cropped_persons"

    folders = [r"/mnt/tanmeng/shelei/data/staff_clothes/D20"]
    out = r"/mnt/tanmeng/shelei/data/staff_clothes/D20/output_pic"
    for folder in folders:
        for root, dirs, files in os.walk(folder):
            if 'frame' in os.path.basename(folder):
                continue
            for filename in files:
                if os.path.splitext(filename)[-1] not in {'.mp4', '.mkv'}:
                    continue
                file_path = os.path.join(root, filename)
                print('start handling video: ', file_path)
                VideoHandle(model_name=None, camera_id=None, video_path=file_path, output=out)
                print('finish handling video: ', file_path)

    print('all done')














