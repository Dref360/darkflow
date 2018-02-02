"""
parse PASCAL VOC xml annotations
"""

import os
from collections import OrderedDict

import h5py

pjoin = os.path.join
import json
import csv
from PIL import Image

import numpy as np
import cv2
import pickle
PATH = "data{}.pkl"

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

class H5Handler():
    def __init__(self,path):
        self.path = path
        self.train_h5 = pjoin(path,'train.h5')
        self.test_h5 = pjoin(path,'test.h5')
        self.trainfile = None
        self.testfile = None

    def __getitem__(self, item):
        is_train,idx = item
        if is_train:
            if self.trainfile is None:
                self.trainfile = h5py.File(self.train_h5, 'r')
            return list(zip(self.trainfile[idx]['mean_angs'],self.trainfile[idx]['mean_mags']))
        if self.testfile is None:
            self.testfile = h5py.File(self.test_h5, 'r')
        return list(zip(self.testfile[idx]['mean_angs'],self.testfile[idx]['mean_mags']))



class JsonHandler():
    classes = ["articulated_truck", "bicycle", "bus", "car", "motorcycle",
               "non-motorized_vehicle", "motorized_vehicle",
               "pedestrian", "pickup_truck", "single_unit_truck", "work_van"]

    def __init__(self, path,by_mio_id=False):
        self.path = path
        self.json = 'internal_cvpr2016.json'
        self.by_mio_id = by_mio_id
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
        if self.by_mio_id:
            return value['mio_id'],[pjoin(self.path, 'images', jpgfile), polygons]
        return pjoin(self.path, 'images', jpgfile), [value['mio_id'], polygons]


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

def area(polygon):
    x, y = [sorted(k) for k in zip(*polygon)]
    # x = [xi / fx for xi in x]
    # y = [yi / fy for yi in y]
    xmin, xmax = x[0], x[-1]
    ymin, ymax = y[0], y[-1]
    return (xmax - xmin) * (ymax - ymin)

def make_bigger(polygon):
    x, y = [sorted(k) for k in zip(*polygon)]
    xmin, xmax = x[0], x[-1]
    ymin, ymax = y[0], y[-1]
    return [(max(0,xmin-10),max(0,ymin-10)),(xmax+10,ymax+10)]

def find_data(handler, ang_getter, is_train):
    for k in handler.gt_train if is_train else handler.gt_test:
        acc = []
        for (cls, polygons), (ang, mag) in zip(handler.datas[k][1], ang_getter[(is_train, k.split('/')[-1][:-4])]):
            acc.append((JsonHandler.classes.index(normalize_classes(cls)), ang, polygons))
        yield (k,acc)

def mio_tcd_loading(ANN, pick, exclusive=False, mode="train"):
    print('Parsing for {} {}'.format(
        pick, mode))

    if os.path.exists(PATH.format(mode)):
        return pickle.load(open(PATH.format(mode),"rb"))

    jsonHandler = JsonHandler(ANN)
    ang_getter = H5Handler(ANN)
    data = find_data(jsonHandler,ang_getter,mode=='train')

    dumps = list()
    for fp, boxes in data:
        all = []
        for cls,ang, cnt in boxes:
            angle = ang / (2*np.pi)
            x_s = sorted([k[0] for k in cnt])
            y_s = sorted([k[1] for k in cnt])

            xn,xx = x_s[0], x_s[-1]
            yn,yx = y_s[0], y_s[-1]

            all.append([cls, xn, yn, xx, yx,angle])
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
    pickle.dump(dumps,open(PATH.format(mode),"wb"))

    return dumps
