#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 06:21:16 2017

@author: zhaoy
"""

import numpy as np
import cv2
import os
import os.path as osp
import sys
import json
import argparse

# reload(sys)
# sys.setdefaultencoding("utf-8")
os.environ['GLOG_minloglevel'] = '2'  # suppress log

import _init_paths
# from matplotlib import pyplot as plt
from mtcnn_aligner import MtcnnAligner

from fx_warp_and_crop_face import get_reference_facial_points, warp_and_crop_face


output_size = (112, 112)
default_square = True
inner_padding_factor = 0
outer_padding = (0, 0)

reference_5pts = get_reference_facial_points(output_size,
                                                inner_padding_factor,
                                                outer_padding,
                                                default_square)

def parse_args():
    parser = argparse.ArgumentParser(description='face alignment by MTCNN')
    # general
    parser.add_argument('--rect-root-dir', default='',
                        help='')
    parser.add_argument('--save-dir', default='./aligned_root_dir',
                        help='')
    parser.add_argument('--image-root-dir', default='',
                        help='')
    parser.add_argument('--mtcnn-model-dir', default='../../model',
                        help='')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='')
    args = parser.parse_args()
    return args

def get_rect_list(rect_list_dir):
    rect_file_list = []
    for root,dirs, files in os.walk(rect_list_dir):
        for name in files:
            if name.endswith(".json"):
                rect_file_list.append(name)
    return rect_file_list


def get_gt_rect(img_info):
    x_center = int(img_info.split(',')[2])
    y_center = int(img_info.split(',')[3])
    w = int(img_info.split(',')[4])
    h = int(img_info.split(',')[5])
    x = x_center - w/2    
    y = y_center - h/2
    rect = [x, y, x+w, y+h]
    return rect


def main(args):

    rect_root_dir = args.rect_root_dir
    save_dir = args.save_dir
    mtcnn_model_dir = args.mtcnn_model_dir
    img_root_dir = args.image_root_dir
    gpu_id = args.gpu_id
 
    rect_list = get_rect_list(rect_root_dir)
    print('all %d images' %(len(rect_list))) 
    if not save_dir:
        save_dir = './aligned_root_dir'

    if not osp.exists(save_dir):
        print('makedirs for aligned root dir: ', save_dir)
        os.makedirs(save_dir)

    save_aligned_dir = osp.join(save_dir, 'aligned_imgs')
    if not osp.exists(save_aligned_dir):
        print('makedirs for aligned/cropped face imgs: ', save_dir)
        os.makedirs(save_aligned_dir)

    save_rects_dir = osp.join(save_dir, 'face_rects')
    if not osp.exists(save_rects_dir):
        print('makedirs for face rects/landmarks: ', save_rects_dir)
        os.makedirs(save_rects_dir)

    aligner = MtcnnAligner(mtcnn_model_dir, True, gpu_id=gpu_id)
    for i in range(len(rect_list)):
        with open(osp.join(rect_root_dir, rect_list[i]), 'r') as f:
            #lines = f.readlines()
            contents = json.load(f)
            img_name = contents['filename']
            gt_rect = contents['faces'][0]['rect']            
            
            if gt_rect is None:
                print('Failed to get_gt_rect(), skip to next image')
                continue

            img = cv2.imread(img_name)
            ht = img.shape[0]
            wd = img.shape[1]

            print 'image.shape:', img.shape
                
            boxes, points = aligner.align_face(img, [gt_rect])

            box = boxes[0]
            pts = points[0]

            spl = img_name.split('/')
            base_name = spl[-1]

            save_img_subdir = osp.join(save_aligned_dir, spl[-2])
            if not osp.exists(save_img_subdir):
                os.makedirs(save_img_subdir)

            save_img_fn = osp.join(save_img_subdir, base_name)
            print('save_img_fn: %s'%save_img_fn)
            facial5points = np.reshape(pts, (2, -1))
            dst_img = warp_and_crop_face(img, facial5points, reference_5pts, output_size)
            cv2.imwrite(save_img_fn, dst_img)

if __name__ == "__main__":
    args = parse_args()
    main(args)
