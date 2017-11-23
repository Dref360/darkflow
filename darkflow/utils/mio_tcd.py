"""
parse PASCAL VOC xml annotations
"""

import os
from collections import OrderedDict

pjoin = os.path.join
import json
import csv
from PIL import Image

import numpy as np
import cv2
import pickle
PATH = "data.pkl"

def _pp(l):  # pretty printing
    for i in l: print('{}: {}'.format(i, l[i]))


def normalize_classes(cls):
    cls = cls.lower()
    dat = {'pickup truck': 'pickup_truck',
           'articulated truck': 'articulated_truck',
           'non-motorized vehicle': 'non-motorized_vehicle',
           'motorized vehicle': 'motorized_vehicle',
           'single unit truck': 'single_unit_truck',
           'work van': 'work_van', 'suv': 'car', 'minivan': 'car'}
    return dat[cls] if cls in dat else cls


class JsonHandler():
    classes = ["articulated_truck", "bicycle", "bus", "car", "motorcycle",
               "non-motorized_vehicle", "motorized_vehicle",
               "pedestrian", "pickup_truck", "single_unit_truck", "work_van"]

    def __init__(self, path):
        self.path = path
        self.json = 'internal_cvpr2016.json'
        file = pjoin(self.path, self.json)
        jsonfile = json.load(open(file, "r"), object_pairs_hook=OrderedDict)
        self.datas = OrderedDict([self.__handle_json_inner(v) for v in jsonfile.values()])
        self.gt_train = 'gt_train.csv'
        self.gt_test = 'gt_test.csv'
        with open(pjoin(self.path, self.gt_train)) as f:
            self.gt_train = list(set([pjoin(self.path, 'images', x[0] + '.jpg') for x in csv.reader(f)]))
        with open(pjoin(self.path, self.gt_test)) as f:
            self.gt_test = set([pjoin(self.path, 'images', x[0] + '.jpg') for x in csv.reader(f)])

    def __handle_json_inner(self, value):
        annotation = value['annotations']
        polygons = [
            (x['classification'], x['outline_xy'])
            for x in annotation]
        jpgfile = value['external_id'] + '.jpg'
        return pjoin(self.path, 'images', jpgfile), polygons


def mio_tcd_loading_regular(ANN, pick, exclusive=False, mode="train"):
    print('Parsing for {} {}'.format(
        pick, mode))

    jsonHandler = JsonHandler(ANN)
    data = jsonHandler.gt_train if mode == "train" else jsonHandler.gt_test
    dumps = list()
    for fp in data:
        all = []
        boxes = jsonHandler.datas[fp]
        for cls, (x, y) in boxes:
            xn, xx = min(x), max(x)
            yn, yx = min(y), max(y)
            cls = normalize_classes(cls)
            name = jsonHandler.classes.index(cls)
            all.append([name, xn, yn, xx, yx])
        im = Image.open(fp)
        dumps.append([fp, [im.width, im.height, all]])

    # gather all stats
    stat = dict()
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current[0] in pick:
                if current[0] in stat:
                    stat[current[0]] += 1
                else:
                    stat[current[0]] = 1

    print('\nStatistics:')
    _pp(stat)
    print('Dataset size: {}'.format(len(dumps)))
    return dumps


def mio_tcd_loading(ANN, pick, exclusive=False, mode="train"):
    print('Parsing for {} {}'.format(
        pick, mode))

    if os.path.exists(PATH):
        return pickle.load(open(PATH,"rb"))

    jsonHandler = JsonHandler(ANN)
    data = jsonHandler.gt_train if mode == "train" else jsonHandler.gt_test
    dumps = list()
    for fp in data:
        all = []
        boxes = jsonHandler.datas[fp]
        for cls, cnt in boxes:
            rect = cv2.minAreaRect(np.array(cnt))
            angle = np.radians(rect[-1])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            xs, ys = list(zip(*box))
            xn, xx = min(xs), max(xs)
            yn, yx = min(ys), max(ys)
            cls = normalize_classes(cls)
            name = jsonHandler.classes.index(cls)
            all.append([name, xn, yn, xx, yx,angle])
        im = Image.open(fp)
        dumps.append([fp, [im.width, im.height, all]])

    # gather all stats
    stat = dict()
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current[0] in pick:
                if current[0] in stat:
                    stat[current[0]] += 1
                else:
                    stat[current[0]] = 1

    print('\nStatistics:')
    _pp(stat)
    print('Dataset size: {}'.format(len(dumps)))
    pickle.dump(dumps,open(PATH,"wb"))

    return dumps
