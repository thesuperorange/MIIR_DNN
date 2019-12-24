# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]

def _get_image_blob(im):
    """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


class FaceDetect():
    def __init__(self):
        self.myargs =self.parse_args()
        lr = cfg.TRAIN.LEARNING_RATE
        momentum = cfg.TRAIN.MOMENTUM
        weight_decay = cfg.TRAIN.WEIGHT_DECAY

        if self.myargs.cfg_file is not None:
            cfg_from_file(self.myargs.cfg_file)

        cfg.USE_GPU_NMS = self.myargs.cuda
        print('Using config:')
        pprint.pprint(cfg)
        np.random.seed(cfg.RNG_SEED)

        self.fasterRCNN = self.training()
        # initilize the tensor holder here.
        im_data = torch.FloatTensor(1)
        im_info = torch.FloatTensor(1)
        num_boxes = torch.LongTensor(1)
        gt_boxes = torch.FloatTensor(1)

        # ship to cuda
        if self.myargs.cuda > 0:
            im_data = im_data.cuda()
            im_info = im_info.cuda()
            num_boxes = num_boxes.cuda()
            gt_boxes = gt_boxes.cuda()

        # make variable
        with torch.no_grad():
            im_data = Variable(im_data)
            im_info = Variable(im_info)
            num_boxes = Variable(num_boxes)
            gt_boxes = Variable(gt_boxes)

        self.im_data = im_data
        self.im_info = im_info
        self.num_boxes = num_boxes
        self.gt_boxes = gt_boxes
        if self.myargs.cuda > 0:
            cfg.CUDA = True



        #start = time.time()
        #max_per_image = 100
        self.thresh = 0.05

    def parse_args(self):
        # --------------arg setting---------------------------
        myargs = Map()
        myargs.dataset = "wider_face"  # pascal_voc
        myargs.cfg_file = 'cfgs/vgg16.yml'
        myargs.net = 'vgg16'  # 'vgg16, res50, res101, res152'
        myargs.arg_set = None  # help='set config keys'
        myargs.load_dir = 'models'  # directory to load models

        myargs.cuda = True
        myargs.mGPU = False
        myargs.class_agnostic = False
        myargs.parallel_type = 0
        myargs.checksession = 1
        myargs.checkepoch = 9  # 4
        myargs.checkpoint = 6439  # 1072
        myargs.bs = False

        myargs.webcam_num = -1
        return myargs

    def training(self):

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.

        input_dir = self.myargs.load_dir + "/" + self.myargs.net + "/" + self.myargs.dataset
        if not os.path.exists(input_dir):
            raise Exception('There is no input directory for loading network from ' + input_dir)
        load_name = os.path.join(input_dir,
                                 'faster_rcnn_{}_{}_{}.pth'.format(
                                     self.myargs.checksession, self.myargs.checkepoch, self.myargs.checkpoint))

        self.pascal_classes = np.asarray(['__background__', 'face'])

        # initilize the network here.
        if self.myargs.net == 'vgg16':
            fasterRCNN = vgg16(self.pascal_classes, pretrained=False, class_agnostic=self.myargs.class_agnostic)
        elif self.myargs.net == 'res101':
            fasterRCNN = resnet(self.pascal_classes, 101, pretrained=False, class_agnostic=self.myargs.class_agnostic)
        elif self.myargs.net == 'res50':
            fasterRCNN = resnet(self.pascal_classes, 50, pretrained=False, class_agnostic=self.myargs.class_agnostic)
        elif self.myargs.net == 'res152':
            fasterRCNN = resnet(self.pascal_classes, 152, pretrained=False, class_agnostic=self.myargs.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()

        fasterRCNN.create_architecture()

        print("load checkpoint %s" % (load_name))
        if self.myargs.cuda > 0:
            checkpoint = torch.load(load_name)
        else:
            checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
        fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']

        print('load model successfully!')

        # pdb.set_trace()

        print("load checkpoint %s" % (load_name))

        if self.myargs.cuda > 0:
            fasterRCNN.cuda()

        fasterRCNN.eval()

        return fasterRCNN
    # ----------------------------------


    def detect(self, dataset, foldername, filename, ch, vis, bbox_log):
        image_num = os.path.splitext(filename)[0]
        output_folder = 'output/' + dataset + "_ch" + str(ch)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        total_tic = time.time()

        # im = cv2.imread(im_file)
        im_file = foldername + "/" + filename

        im_in = np.array(imread(im_file))

        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)
        # rgb -> bgr
        im = im_in[:, :, ::-1]

        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)


        with torch.no_grad():
            self.im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            self.im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            self.gt_boxes.resize_(1, 1, 5).zero_()
            self.num_boxes.resize_(1).zero_()

        # pdb.set_trace()
        det_tic = time.time()

        rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, rois_label = self.fasterRCNN(
            self.im_data, self.im_info, self.gt_boxes, self.num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if self.myargs.class_agnostic:
                    if self.myargs.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if self.myargs.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4 * len(self.pascal_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, self.im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im2show = np.copy(im)
        for j in xrange(1, len(self.pascal_classes)):
            inds = torch.nonzero(scores[:, j] > self.thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if self.myargs.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]

                if bbox_log:
                    bbox_list = cls_dets.cpu().numpy()
                    for bb in bbox_list:
                        start_x = int(bb[0])
                        start_y = int(bb[1])
                        end_x = int(bb[2])
                        end_y = int(bb[3])
                        confidence = bb[4]
                        if confidence > 0.5:
                            fo.write(
                                str(ch) + "," + image_num + "," + str(start_x) + "," + str(start_y) + "," +
                                str(end_x) + "," + str(end_y) + "," + str(confidence) + "\n"
                            )

                if vis:
                    im2show = vis_detections(im2show, self.pascal_classes[j], cls_dets.cpu().numpy(), 0.5)

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        # sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
        #                       .format(num_images + 1, len(imglist), detect_time, nms_time))
        # sys.stdout.flush()
        if vis:
            result_path = os.path.join(output_folder, str(image_num) + ".jpg")
            cv2.imwrite(result_path, im2show)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='Pathway1_1', help='dataset')
    #    parser.add_argument('-f', '--frame_num', default=548, help='total frame')
    parser.add_argument('-s', '--savefig', action='store_true')
    parser.add_argument('-l', '--savelog', action='store_true')
    parser.add_argument('-i', '--input_path', default='MI3', help='input image path')
    #    parser.add_argument('-i', '--detect_face', action='store_true')
    #    parser.add_argument('-e', '--ext',  default='png', help='png,bmp,jpg')
    #    parser.add_argument('-b', '--begin_frame', default = 0, help='set start frame')
    #    parser.add_argument('--threshold',default = 0.8,help='confidence threshold')
    args = parser.parse_args()

    vis = args.savefig
    bbox_log = args.savelog
    dataset = args.dataset
    channel_list = [2, 4, 6]
    MI3path = args.input_path

    fo = open("faster-rcnn_face_" + dataset + ".txt", "w")

    fd = FaceDetect()
    for ch in channel_list:
        input_folder = os.path.join(MI3path, dataset, 'ORIG/ch' + str(ch))
        for filename in os.listdir(input_folder):
            fd.detect(dataset, input_folder, filename, ch, vis, bbox_log)

    fo.close()


