#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# -*- coding: utf-8 -*- 
import os
import time
import json
import cv2
import numpy as np
import argparse
import glob
import sophon.sail as sail
import logging
logging.basicConfig(level=logging.INFO)

class Test(object):
    def __init__(self, args):
        # load bmodel
        print("load model!") 

      
    def __call__(self, img_list):
        print("start!!!") 

def main(args):
    # creat save path
    filename_list = []

    for filename in glob.glob(args.input+'/*'):
        if os.path.splitext(filename)[-1] not in ['.jpg','.png','.jpeg','.bmp','.JPEG','.JPG','.BMP']:
            continue
        
        src_img = cv2.imread(filename)
        print("open :"+filename)
        if src_img is None:
            logging.error("{} imread is None.".format(filename))
            continue
        
        filename = os.path.basename(filename)
        filename_list.append(filename)

        
def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='datasets/', help='path of input, must be image directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argsparser()
    main(args)
