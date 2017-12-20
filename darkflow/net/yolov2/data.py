from ...utils.pascal_voc_clean_xml import pascal_voc_clean_xml
from numpy.random import permutation as perm
from ..yolo.predict import preprocess
from ..yolo.data import shuffle
from copy import deepcopy
from .train import USE_REG
import pickle
import numpy as np
import os




def _batch_cat(self, chunk):
    """
    Takes a chunk of parsed annotations
    returns value for placeholders of net's
    input & loss layer correspond to this chunk
    """
    meta = self.meta
    labels = meta['labels']
    n_dim = 8
    gt = np.identity(n_dim)
    H, W, _ = meta['out_size']
    C, B = meta['classes'], meta['num']
    anchors = meta['anchors']

    # preprocess
    jpg = chunk[0];
    w, h, allobj_ = chunk[1]
    allobj = deepcopy(allobj_)
    path = jpg
    img = self.preprocess(path, allobj)

    # Calculate regression target
    cellx = 1. * w / W
    celly = 1. * h / H
    for obj in allobj:
        centerx = .5 * (obj[1] + obj[3])  # xmin, xmax
        centery = .5 * (obj[2] + obj[4])  # ymin, ymax
        cx = centerx / cellx
        cy = centery / celly
        if cx >= W or cy >= H: return None, None
        obj[3] = float(obj[3] - obj[1]) / w
        obj[4] = float(obj[4] - obj[2]) / h
        obj[3] = np.sqrt(obj[3])
        obj[4] = np.sqrt(obj[4])
        obj[1] = cx - np.floor(cx)  # centerx
        obj[2] = cy - np.floor(cy)  # centery
        obj += [int(np.floor(cy) * W + np.floor(cx))]

    # show(im, allobj, S, w, h, cellx, celly) # unit test

    # Calculate placeholders' values
    probs = np.zeros([H * W, B, C])
    confs = np.zeros([H * W, B])
    coord = np.zeros([H * W, B, 4])
    proid = np.zeros([H * W, B, C])
    prear = np.zeros([H * W, 4])
    thetas = np.zeros([H * W, B,n_dim])
    for obj in allobj:
        probs[obj[6], :, :] = [[0.] * C] * B
        probs[obj[6], :, obj[0]] = 1.
        proid[obj[6], :, :] = [[1.] * C] * B
        coord[obj[6], :, :] = [obj[1:5]] * B
        prear[obj[6], 0] = obj[1] - obj[3] ** 2 * .5 * W  # xleft
        prear[obj[6], 1] = obj[2] - obj[4] ** 2 * .5 * H  # yup
        prear[obj[6], 2] = obj[1] + obj[3] ** 2 * .5 * W  # xright
        prear[obj[6], 3] = obj[2] + obj[4] ** 2 * .5 * H  # ybot
        confs[obj[6], :] = [1.] * B

        ## ANGLE
        angle_cls = int(((np.rad2deg(obj[5]) + 90) % 180) // (180/n_dim))
        thetas[obj[6], :,angle_cls] = 1.0

    # Finalise the placeholders' values
    upleft = np.expand_dims(prear[:, 0:2], 1)
    botright = np.expand_dims(prear[:, 2:4], 1)
    wh = botright - upleft;
    area = wh[:, :, 0] * wh[:, :, 1]
    upleft = np.concatenate([upleft] * B, 1)
    botright = np.concatenate([botright] * B, 1)
    areas = np.concatenate([area] * B, 1)
    # thetas = np.expand_dims(thetas,-1)

    # value for placeholder at input layer
    inp_feed_val = img
    # value for placeholder at loss layer
    loss_feed_val = {
        'probs': probs, 'confs': confs,
        'coord': coord, 'proid': proid,
        'areas': areas, 'upleft': upleft,
        'botright': botright, 'thetas': thetas
    }

    return inp_feed_val, loss_feed_val


def _batch_reg(self, chunk):
    """
    Takes a chunk of parsed annotations
    returns value for placeholders of net's 
    input & loss layer correspond to this chunk
    """
    meta = self.meta
    labels = meta['labels']
    
    H, W, _ = meta['out_size']
    C, B = meta['classes'], meta['num']
    anchors = meta['anchors']

    # preprocess
    jpg = chunk[0]; w, h, allobj_ = chunk[1]
    allobj = deepcopy(allobj_)
    path = jpg
    img = self.preprocess(path, allobj)

    # Calculate regression target
    cellx = 1. * w / W
    celly = 1. * h / H
    for obj in allobj:
        centerx = .5*(obj[1]+obj[3]) #xmin, xmax
        centery = .5*(obj[2]+obj[4]) #ymin, ymax
        cx = centerx / cellx
        cy = centery / celly
        if cx >= W or cy >= H: return None, None
        obj[3] = float(obj[3]-obj[1]) / w
        obj[4] = float(obj[4]-obj[2]) / h
        obj[3] = np.sqrt(obj[3])
        obj[4] = np.sqrt(obj[4])
        obj[1] = cx - np.floor(cx) # centerx
        obj[2] = cy - np.floor(cy) # centery
        obj += [int(np.floor(cy) * W + np.floor(cx))]

    # show(im, allobj, S, w, h, cellx, celly) # unit test

    # Calculate placeholders' values
    probs = np.zeros([H*W,B,C])
    confs = np.zeros([H*W,B])
    coord = np.zeros([H*W,B,4])
    proid = np.zeros([H*W,B,C])
    prear = np.zeros([H*W,4])
    thetas = np.zeros([H * W, B])
    for obj in allobj:
        probs[obj[6], :, :] = [[0.]*C] * B
        probs[obj[6], :, obj[0]] = 1.
        proid[obj[6], :, :] = [[1.]*C] * B
        coord[obj[6], :, :] = [obj[1:5]] * B
        prear[obj[6],0] = obj[1] - obj[3]**2 * .5 * W # xleft
        prear[obj[6],1] = obj[2] - obj[4]**2 * .5 * H # yup
        prear[obj[6],2] = obj[1] + obj[3]**2 * .5 * W # xright
        prear[obj[6],3] = obj[2] + obj[4]**2 * .5 * H # ybot
        confs[obj[6], :] = [1.] * B
        thetas[obj[6], :] = [obj[5]] * B

    # Finalise the placeholders' values
    upleft   = np.expand_dims(prear[:,0:2], 1)
    botright = np.expand_dims(prear[:,2:4], 1)
    wh = botright - upleft; 
    area = wh[:,:,0] * wh[:,:,1]
    upleft   = np.concatenate([upleft] * B, 1)
    botright = np.concatenate([botright] * B, 1)
    areas = np.concatenate([area] * B, 1)
    #thetas = np.expand_dims(thetas,-1)

    # value for placeholder at input layer
    inp_feed_val = img
    # value for placeholder at loss layer 
    loss_feed_val = {
        'probs': probs, 'confs': confs, 
        'coord': coord, 'proid': proid,
        'areas': areas, 'upleft': upleft, 
        'botright': botright, 'thetas':thetas
    }

    return inp_feed_val, loss_feed_val

_batch = _batch_reg if USE_REG else _batch_cat
