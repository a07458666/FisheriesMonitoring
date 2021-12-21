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
            label = 0
#             label = label_map[ann["class"]]
            x = ann["x"]
            y = ann["y"]
            height = ann["height"]
            width = ann["width"]
#             print(label)
#             print(x)
#             print(y)
#             print(height)
#             print(width)
#             print(image_name)
            h,w,c = get_image_size(image_name)
#             print(h,w,c)
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
            
#         for im in images:
#             dw = 1. / im['width']
#             dh = 1. / im['height']

#             annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
#             anns = coco.loadAnns(annIds)

#             filename = im['file_name'].replace(".jpg", ".txt")
#             print(filename)   
#             with open("labels/" + filename, "a") as myfile:
#                 for i in range(len(anns)):
#                     xmin = anns[i]["bbox"][0]
#                     ymin = anns[i]["bbox"][1]
#                     xmax = anns[i]["bbox"][2] + anns[i]["bbox"][0]
#                     ymax = anns[i]["bbox"][3] + anns[i]["bbox"][1]

#                     x = (xmin + xmax)/2
#                     y = (ymin + ymax)/2

#                     w = xmax - xmin
#                     h = ymax-ymin

#                     x = x * dw
#                     w = w * dw
#                     y = y * dh
#                     h = h * dh

#                     # Note: This assumes a single-category dataset, and thus the "0" at the beginning of each line.
#                     mystring = str("0 " + str(truncate(x, 7)) + " " + str(truncate(y, 7)) + " " + str(truncate(w, 7)) + " " + str(truncate(h, 7)))
#                     myfile.write(mystring)
#                     myfile.write("\n")

# with h5py.File(PATH_MAT) as hdf5_data:
#     for i in tqdm(range(33402)):
#         img_name = get_name(i, hdf5_data)
#         if not os.path.isfile(DATA_PATH_TRAIN + img_name):
#             continue
#         im = cv2.imread(DATA_PATH_TRAIN + img_name)
#         h, w, c = im.shape
#         fp = open(DATA_PATH_TRAIN + img_name.replace(".png", ".txt"), "w")
#         arr = get_bbox(i, hdf5_data)
#         #         print(arr)
#         arr_l = len(arr["label"])
#         annotations = []
#         for idx in range(arr_l):
#             label = arr["label"][idx]
#             if label == 10:
#                 label = 0
#             _l = arr["left"][idx]
#             _t = arr["top"][idx]
#             _w = arr["width"][idx]
#             if (_l + _w) > w:
#                 _w = w - _l - 1
#             _h = arr["height"][idx]
#             if (_t + _h) > h:
#                 _h = h - _t - 1
#             x_center = (_l + _w / 2) / w
#             y_center = (_t + _h / 2) / h
#             bbox_width = _w / w
#             bbox_height = _h / h
#             start_point = (
#                 int(w * (x_center - (bbox_width / 2))),
#                 int(h * (y_center - (bbox_height / 2))),
#             )
#             end_point = (
#                 int(w * (x_center + (bbox_width / 2))),
#                 int(h * (y_center + (bbox_height / 2))),
#             )
#             im = cv2.rectangle(im, start_point, end_point, color, thickness)
#             s = (
#                 str(label)
#                 + " "
#                 + str(x_center)
#                 + " "
#                 + str(y_center)
#                 + " "
#                 + str(bbox_width)
#                 + " "
#                 + str(bbox_height)
#             )
#             if idx != (arr_l - 1):
#                 s += "\n"
#             fp.write(s)
#         fp.close()
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         plt.imshow(im)
#         plt.title(img_name)
#         plt.show()
