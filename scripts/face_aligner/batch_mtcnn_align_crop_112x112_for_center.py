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
import math
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


def main(args):

    rect_root_dir = args.rect_root_dir
    save_dir = args.save_dir
    mtcnn_model_dir = args.mtcnn_model_dir
    img_root_dir = args.image_root_dir
    gpu_id = args.gpu_id
    MAX_LINE = 500 
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
            face_count = contents['face_count']        
            if face_count == 0:
                print('not detect face: %s'%img_name)
                continue
       
            img = cv2.imread(img_name)
            img_h = img.shape[0]
            img_w = img.shape[1]


            max_line = 0
            resize_scale = 0
            new_img_h = img_h
            new_img_w = img_w
            if img_h >= img_w:
                max_line = img_h
                if max_line > MAX_LINE:
                    resize_scale = max_line / MAX_LINE
                    new_img_h = MAX_LINE
                    new_img_w = img_w / resize_scale

            if img_w > img_h:
                max_line = img_w
                if max_line > MAX_LINE:
                    resize_scale = max_line / MAX_LINE
                    new_img_w = MAX_LINE
                    new_img_h = img_h / resize_scale
            print 'image.shape:', img.shape
            img = cv2.resize(img, (new_img_w, new_img_h))
            img_center_x = img_w/2
            img_center_y = img_h/2
            #choose center face
            center_face_index = 0

            min_dist = 9999999
            gt_rect = ''
            for j in range(face_count):
                 gt_rect = contents['faces'][j]['rect']
                 face_center_x = int(gt_rect[0]) + int(gt_rect[2] / 2)
                 face_center_y = int(gt_rect[1]) + int(gt_rect[3] / 2)
                 dist = math.sqrt(math.pow(face_center_x - img_center_x, 2) + math.pow(face_center_y - img_center_y, 2))
                 if dist < min_dist:
                     min_dist = dist
                     center_face_index = j

            gt_rect = contents['faces'][center_face_index]['rect']    

            
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
