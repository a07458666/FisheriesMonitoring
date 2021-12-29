import numpy as np
import os
import json
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

# Fish_Dataset
PATH_ANN = "../rectangle/"
PATH_IMAGE = "../../../DATA/fish/train_all/"
label_map = {'ALB':0,'BET':1,'DOL':2,'LAG':3,'Nof':4,'OTHER':5,'SHARK':6,'YFT':7}

def get_image_size(image_name):
    im = cv2.imread(PATH_IMAGE + image_name)
    h, w, c = im.shape
    return h,w,c

def jsonToTxt(data):
    for d in tqdm(data):
        anns = d['annotations']
        image_name = d['filename'].split("/")[-1]
        fp = open(PATH_IMAGE + image_name.replace(".jpg", ".txt"), "w")
        for ann in anns:
            label = label_map[ann["class"]]
            x = ann["x"]
            y = ann["y"]
            height = ann["height"]
            width = ann["width"]

            h,w,c = get_image_size(image_name)
            x_center = (x + width / 2) / w
            y_center = (y + height / 2) / h
            bbox_width = width / w
            bbox_height = height / h
            s = (str(label)+ " " + str(x_center) + " " + str(y_center) + " " + str(bbox_width) + " " + str(bbox_height))
            if ann != anns[-1]:
                s += "\n"
            fp.write(s)
        fp.close()

for fileName in os.listdir(PATH_ANN):
    if fileName.endswith(('.json')):
        path = os.path.join(PATH_ANN, fileName)
        with open(path) as f:
            data = json.load(f)
            jsonToTxt(data)