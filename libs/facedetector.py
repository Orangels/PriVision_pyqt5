# -*- coding:utf8 -*-
import sys

sys.path.append('/home/user/workspace/openpose-priv-dev/3rdparty/caffe/python')
sys.path.append('/home/user/workspace/openpose-priv-dev/3rdparty/lib/priv_tools')

import caffe
import label_info
from inference import Detector
from common import objs_sort_by_center
from config import cfg


class FaceDetector:
    def __init__(self, gpu_id=0):
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        det_net = caffe.Net(cfg.FaceDet.DEPLOY, cfg.FaceDet.WEIGHTS, caffe.TEST)

        self.D = Detector(det_net, mean=cfg.FaceDet.PIXEL_MEANS, std=cfg.FaceDet.PIXEL_STDS,
                          scales=(480,), max_sizes=(800,), preN=500, postN=50, conf_thresh=0.7,
                          color_map={0: [192, 0, 192], 1: [255, 64, 64]}, class_map=label_info.FACE_CLASS)

    def __call__(self, img):
        objs = self.D.det_im(img)
        if objs is None:
            return None
        else:
            return objs_sort_by_center(objs)

    def __del__(self):
        print self.__class__.__name__

