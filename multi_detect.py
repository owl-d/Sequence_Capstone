# Mecab 설치 코드
# !sudo apt-get install g++ openjdk-8-jdk python3-dev python3-pip curl
# !python3 -m pip install --upgrade pip
# !python3 -m pip install konlpy
# !sudo apt-get install curl git
# !bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)

import time
from flask import Flask, request, jsonify
import base64
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import json
import shutil
from glob import glob
from pathlib import Path
import sys
import yaml
from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import Annotator
from utils.torch_utils import select_device, time_sync
from Google_Cloud_Vision_API import GCV_model, CU, GS, Emart24
from konlpy.tag import Mecab
#from rembg.bg import remove as remove_bg


json_class_path = './classify.json'
json_label_path = './label2name.json'
with open(json_class_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)
with open(json_label_path, 'r', encoding='utf-8') as f:
    json_label = json.load(f)

yolo_ready_flag = True
object_detect_flag = False
WEIGHTS = '640_s_model.pt'
IMG_SIZE = 640
DEVICE = ''
AUGMENT = False
CONF_THRES = 0.25
IOU_THRES = 0.45
CLASSES = None
AGNOSTIC_NMS = False
target_object = "Init"
target_category = "Init"
Input_ETRI = []

app = Flask(__name__)

device = select_device(DEVICE)
model = attempt_load(WEIGHTS, map_location=device)  # load FP32 model


@app.route('/', methods=['GET', 'POST'])
def detect():
    global yolo_ready_flag
    global object_detect_flag
    global WEIGHTS
    global IMG_SIZE
    global DEVICE
    global AUGMENT
    global CONF_THRES
    global IOU_THRES
    global CLASSES
    global AGNOSTIC_NMS
    global target_object
    global target_category
    global Input_ETRI

    if request.method == 'POST':
        box_ = []
        labels_ = []
        max_name = []
        _returns = {'start' : 'blank'}

        data = request.data

        ####### etri target receive ############################
        if data[:4] == b'etri':
            target_object = data.decode('utf-8')
            target_object = target_object[4:]
            if (target_object[-1] == "."):
                target_object = target_object[:-1]

            # if space split
            #target_object = target_object.split(' ')
            # if no split str (사실 둘 다 적용 가능)
            mecab = Mecab()
            target_object = mecab.nouns(target_object.replace(" ",""))

            Out_Category = []
            Input_ETRI = target_object
            for ETRI in Input_ETRI:
                for key in json_data.keys():
                    for value in json_data[key]:
                        if ETRI in value:
                            Out_Category.append(key)

            target_category = Out_Category
            for obj, cate in zip(target_object,target_category):
                print("target_object : " + obj)
                print("target_category :" + cate)
            target_object = ' '.join([obj for obj in target_object])
            target_category = ' '.join([cate for cate in target_category])

            return "1"
        ########################################################

        ####### smart bill #####################################
        if data[:4] == b'bill':
            print("bill")
            img_str = data[4:]
            decoded_string = np.fromstring(base64.b64decode(img_str), np.uint8)
            #print(img_str[900:905])
            img = cv2.imdecode(decoded_string, cv2.IMREAD_COLOR)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite('from_android_bill.jpg', img)

            with open('from_android_bill.jpg', 'rb') as f:
                content = f.read()

            save_path = "GCV_text.txt"
            GCV_model(content,save_path)
            with open(save_path, "r+", encoding="UTF-8") as file:
                strings = file.readlines()
            digit, text = CU(Input_ETRI,strings)
            #GS(target_object,strings)
            #Emart24(target_object,strings)
            file.close()

            print(digit)
            print(text)
            if len(text) != 0:
                return "2"
            else:
                return "7" + ' '.join(text)
        ########################################################

        if yolo_ready_flag == True:

            yolo_ready_flag = False

            weights, imgsz = WEIGHTS, IMG_SIZE

            # Initialize
            #device = select_device(DEVICE)
            half = device.type != 'cpu'  # half precision only supported on CUDA
            # print('device:', device)

            # Load model
            #model = attempt_load(weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check img_size
            if half:
                model.half()  # to FP16

            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

            # Load image
            ### Decode base64 ################################
            image_str = data
            decoded_string = np.fromstring(base64.b64decode(image_str), np.uint8)
            #print(image_str[900:905])
            img0 = cv2.imdecode(decoded_string, cv2.IMREAD_COLOR)
            img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite('from_android_yolo.jpg', img0)
            img0 = cv2.resize(img0, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            #img0 = remove_bg(img0)
            # assert img0 is not None, 'Image Not Found ' + source

            # Padded resize
            img = letterbox(img0, imgsz, stride=stride)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t0 = time_sync()
            pred = model(img, augment=AUGMENT)[0]
            # print('pred shape:', pred.shape)

            # Apply NMS
            pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)

            # Process detections
            det = pred[0]
            # print('det shape:', det.shape)

            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string

            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            labels = []
            if len(det):
                # Rescale boxes from img_size to img0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # if os.path.isfile(txt_path + '.txt'):
                #     os.remove(txt_path + '.txt')

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf)
                    box_.append(xywh)
                    # with open(txt_path + '.txt', 'a') as f:
                    #     f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # label = f'{names[int(cls)]} {conf:.2f}'
                    label = names[int(cls)]
                    labels.append(label)

                # print(f'Inferencing and Processing Done. ({time.time() - t0:.3f}s)')

            for idx in labels:
                labels_.append(json_label[str(idx)])
            _len_list = []
            for index in list(json_data.keys()):
                intersection = [index for idx in labels_ if len(list(set(json_data[str(index)]) & {idx}))]
                _len_list.append(len(intersection))
            max_name.append(list(json_data.keys())[_len_list.index(max(_len_list))])


            yolo_ready_flag = True
            #print("Bounding Box :", box_)
            print("현재 카테고리 :", max_name)
            print("현재 품목 :", labels_)

            object_list = target_object.split(' ')
            category_list = target_category.split(' ')
            if ((object_detect_flag == False) and max_name[0] in category_list): #Find Category?

                ### Setting for Object Detecting ######
                #WEIGHTS = 'back_320small.pt'
                #IMG_SIZE = 320
                object_detect_flag = True
                #######################################
                print("Find Target Category")
                if max_name[0] == "유제품":
                    return "10"
                elif max_name[0] == "과자":
                    return "11"
                elif max_name[0] == "음료":
                    return "12"
                return "3"


            elif (object_detect_flag == True): #Find Object After Find Category?

                for obj in object_list:
                    for i in labels_:
                        # print(obj, labels_)
                        if (i.find(obj) != -1):
                            print("Find Target Object")
                            if i == "남양맛있는우유GT":
                                return "20"
                            elif i == "농심매운새우깡90G":
                                return "21"
                            elif i == "매일아몬드브리즈뉴트리플러스프로틴":
                                return "22"
                            elif i == "코카콜라)코카콜라350ML":
                                return "23"
                            elif i == "농심자갈치90G":
                                return "24"
                            elif i == "농심바나나킥75G":
                                return "25"
                            elif i == "코카토레타500ML":
                                return "26"
                            elif i == "서울우유":
                                return "27"
                            elif i == "코카환타오렌지1.5L":
                                return "28"

                if (max_name[0] in category_list): #Cannot Find Object? So, Check Max_Category
                    print("Find Target Category But Not Object")
                    if max_name[0] == "유제품":
                        return "10"
                    elif max_name[0] == "과자":
                        return "11"
                    elif max_name[0] == "음료":
                        return "12"
                    return "3"

                else:
                    ### Setting to Return Category Detecting #####
                    #WEIGHTS = 'back_small.pt'
                    #IMG_SIZE = 640
                    object_detect_flag = False
                    ##############################################
        print("Detecting ...")
        return "6"

    else:
        return "GET or else"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
