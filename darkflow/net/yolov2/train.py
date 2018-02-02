import tensorflow as tf
import numpy as np

def expit_tensor(x):
    return 1. / (1. + tf.exp(-x))


USE_REG = True

def dot(a, b):
    """Real dot-product for vectors (batch-size,?)"""
    return tf.reduce_sum(tf.multiply(a, b), -1, True)

def dotloss(y_true, y_pred):
    """
    Dot-product loss between 2 angles
    Args:
        y_true: [?,1] the true angle (between 0 and 1)
        y_pred: [?,1] the predicted angle (between 0 and 1)

    Returns: 1 - dotproduct(v1,v2)

    """
    a = y_true * 2 * np.pi
    b = y_pred * 2 * np.pi

    av = tf.stack([tf.cos(a), tf.sin(a)], -1)
    bv = tf.stack([tf.cos(b), tf.sin(b)], -1)
    return 1.0 - tf.reduce_sum(dot(av, bv), axis=-1)


def arctan_loss(y_true, y_pred):
    a = y_true * 2 * np.pi
    b = y_pred * 2 * np.pi
    return tf.reduce_mean(tf.square(tf.atan2(tf.sin(a - b), tf.cos(a - b))), axis=-1)


def loss_reg(self, net_out):
    """
    Takes net.out and placeholders value
    returned in batch() func above,
    to build train_op and loss
    """
    # meta
    m = self.meta
    sprob = float(m['class_scale'])
    sconf = float(m['object_scale'])
    snoob = float(m['noobject_scale'])
    scoor = float(m['coord_scale'])
    H, W, _ = m['out_size']
    B, C = m['num'], m['classes']
    HW = H * W  # number of grid cells
    anchors = m['anchors']

    print('{} loss hyper-parameters:'.format(m['model']))
    print('\tH       = {}'.format(H))
    print('\tW       = {}'.format(W))
    print('\tbox     = {}'.format(m['num']))
    print('\tclasses = {}'.format(m['classes']))
    print('\tscales  = {}'.format([sprob, sconf, snoob, scoor]))

    size1 = [None, HW, B, C]
    size2 = [None, HW, B]

    # return the below placeholders
    _probs = tf.placeholder(tf.float32, size1)
    _confs = tf.placeholder(tf.float32, size2)
    _coord = tf.placeholder(tf.float32, size2 + [4])
    # weights term for L2 loss
    _proid = tf.placeholder(tf.float32, size1)
    # material calculating IOU
    _areas = tf.placeholder(tf.float32, size2)
    _upleft = tf.placeholder(tf.float32, size2 + [2])
    _botright = tf.placeholder(tf.float32, size2 + [2])
    _theta = tf.placeholder(tf.float32, size2 + [1])

    self.placeholders = {
        'probs': _probs, 'confs': _confs, 'coord': _coord, 'proid': _proid,
        'areas': _areas, 'upleft': _upleft, 'botright': _botright, 'thetas': _theta
    }

    # Extract the coordinate prediction from net.out
    net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (4 + 1 + C + 1)])
    coords = net_out_reshape[:, :, :, :, :4]
    coords = tf.reshape(coords, [-1, H * W, B, 4])


    # TODO maybe tanh?
    theta_pred = tf.reshape(tf.sigmoid(net_out_reshape[:, :, :, :, -1]),[-1,19*19,5,1])
    adjusted_coords_xy = expit_tensor(coords[:, :, :, 0:2])
    adjusted_coords_wh = tf.sqrt(tf.exp(coords[:, :, :, 2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]))
    coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

    adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 4])
    adjusted_c = tf.reshape(adjusted_c, [-1, H * W, B, 1])

    adjusted_prob = tf.nn.tanh(net_out_reshape[:, :, :, :, 5:-1])
    adjusted_prob = tf.reshape(adjusted_prob, [-1, H * W, B, C])

    adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

    wh = tf.pow(coords[:, :, :, 2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])
    area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]
    centers = coords[:, :, :, 0:2]
    floor = centers - (wh * .5)
    ceil = centers + (wh * .5)

    # calculate the intersection areas
    intersect_upleft = tf.maximum(floor, _upleft)
    intersect_botright = tf.minimum(ceil, _botright)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = tf.multiply(intersect_wh[:, :, :, 0], intersect_wh[:, :, :, 1])

    # calculate the best IOU, set 0.0 confidence for worse boxes
    iou = tf.truediv(intersect, _areas + area_pred - intersect)
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
    best_box = tf.to_float(best_box)
    confs = tf.multiply(best_box, _confs)
    #theta_pred = tf.expand_dims(tf.multiply(tf.reshape(theta_pred,[-1,19*19,B]),confs),-1)

    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs
    weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
    cooid = scoor * weight_coo
    weight_pro = tf.concat(C * [tf.expand_dims(confs, -1)], 3)
    proid = sprob * weight_pro

    self.fetch += [_probs, confs, conid, cooid, proid]
    true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs], 3)
    wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid], 3)

    print('Building {} loss'.format(m['model']))
    loss = tf.pow(adjusted_net_out - true, 2)
    loss = tf.multiply(loss, wght)
    loss = tf.reshape(loss, [-1, H * W * B * (4 + 1 + C)])
    loss = tf.reduce_sum(loss, 1)
    L_theta = dotloss(_theta,theta_pred)
    self.loss = .5 * tf.reduce_mean(loss)
    tf.summary.scalar('{} box loss'.format(m['model']), self.loss)
    angle_loss = 0.5 * tf.reduce_mean(L_theta * tf.expand_dims(_confs,-1))
    tf.summary.scalar('{} angle loss'.format(m['model']), self.loss)
    self.loss += angle_loss
    tf.summary.scalar('{} loss'.format(m['model']), self.loss)


def loss_cat(self, net_out):
    """
    Takes net.out and placeholders value
    returned in batch() func above,
    to build train_op and loss
    """
    # meta
    m = self.meta
    sprob = float(m['class_scale'])
    sconf = float(m['object_scale'])
    snoob = float(m['noobject_scale'])
    scoor = float(m['coord_scale'])
    H, W, _ = m['out_size']
    B, C = m['num'], m['classes']
    HW = H * W  # number of grid cells
    anchors = m['anchors']

    print('{} loss hyper-parameters:'.format(m['model']))
    print('\tH       = {}'.format(H))
    print('\tW       = {}'.format(W))
    print('\tbox     = {}'.format(m['num']))
    print('\tclasses = {}'.format(m['classes']))
    print('\tscales  = {}'.format([sprob, sconf, snoob, scoor]))

    size1 = [None, HW, B, C]
    size2 = [None, HW, B]

    # return the below placeholders
    _probs = tf.placeholder(tf.float32, size1)
    _confs = tf.placeholder(tf.float32, size2)
    _coord = tf.placeholder(tf.float32, size2 + [4])
    # weights term for L2 loss
    _proid = tf.placeholder(tf.float32, size1)
    # material calculating IOU
    _areas = tf.placeholder(tf.float32, size2)
    _upleft = tf.placeholder(tf.float32, size2 + [2])
    _botright = tf.placeholder(tf.float32, size2 + [2])
    _theta = tf.placeholder(tf.float32, size2 + [8])

    self.placeholders = {
        'probs': _probs, 'confs': _confs, 'coord': _coord, 'proid': _proid,
        'areas': _areas, 'upleft': _upleft, 'botright': _botright, 'thetas': _theta
    }

    # Extract the coordinate prediction from net.out
    net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (4 + 1 + C + 8)])
    coords = net_out_reshape[:, :, :, :, :4]
    coords = tf.reshape(coords, [-1, H * W, B, 4])
    theta_pred = net_out_reshape[:, :, :, :, -8:]
    adjusted_coords_xy = expit_tensor(coords[:, :, :, 0:2])
    adjusted_coords_wh = tf.sqrt(tf.exp(coords[:, :, :, 2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]))
    coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

    adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 4])
    adjusted_c = tf.reshape(adjusted_c, [-1, H * W, B, 1])

    adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:5 + C])
    adjusted_prob = tf.reshape(adjusted_prob, [-1, H * W, B, C])

    adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

    wh = tf.pow(coords[:, :, :, 2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])
    area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]
    centers = coords[:, :, :, 0:2]
    floor = centers - (wh * .5)
    ceil = centers + (wh * .5)

    # calculate the intersection areas
    intersect_upleft = tf.maximum(floor, _upleft)
    intersect_botright = tf.minimum(ceil, _botright)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = tf.multiply(intersect_wh[:, :, :, 0], intersect_wh[:, :, :, 1])

    # calculate the best IOU, set 0.0 confidence for worse boxes
    iou = tf.truediv(intersect, _areas + area_pred - intersect)
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
    best_box = tf.to_float(best_box)
    confs = tf.multiply(best_box, _confs)

    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs
    weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
    cooid = scoor * weight_coo
    weight_pro = tf.concat(C * [tf.expand_dims(confs, -1)], 3)
    proid = sprob * weight_pro

    self.fetch += [_probs, confs, conid, cooid, proid]
    true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs], 3)
    wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid], 3)

    print('Building {} loss'.format(m['model']))
    loss = tf.pow(adjusted_net_out - true, 2)
    loss = tf.multiply(loss, wght)
    loss = tf.reshape(loss, [-1, H * W * B * (4 + 1 + C)])
    loss = tf.reduce_sum(loss, 1)
    L_theta = tf.nn.softmax_cross_entropy_with_logits(labels=_theta, logits=theta_pred)
    self.loss = .5 * tf.reduce_mean(loss)
    tf.summary.scalar('{} box loss'.format(m['model']), self.loss)
    angle_loss = 100 * tf.reduce_mean(tf.reshape(L_theta, [-1, 19 * 19, B]) * ((0.75 * _confs) + 0.25 * (1 - _confs)))
    tf.summary.scalar('{} angle loss'.format(m['model']), self.loss)
    self.loss += angle_loss
    tf.summary.scalar('{} loss'.format(m['model']), self.loss)


if USE_REG:
    loss = loss_reg
else:
    loss = loss_cat
