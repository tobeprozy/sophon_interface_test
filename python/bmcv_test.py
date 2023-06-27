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
import copy


class Test(object):
    def __init__(self, args):
        # self.handle = self.net.get_handle()
        self.handle = sail.Handle(args.dev_id)
        self.bmcv = sail.Bmcv(self.handle)


def main(args):
    # initialize net
    test = Test(args)

    filename_list = []

    for filename in glob.glob(args.input+'/*'):
        if os.path.splitext(filename)[-1] not in ['.jpg','.png','.jpeg','.bmp','.JPEG','.JPG','.BMP']:
            continue
       
        decoder = sail.Decoder(filename, True, args.dev_id)
        bmimg = sail.BMImage()
        ret = decoder.read(test.handle, bmimg)
        if ret != 0:
            logging.error("{} decode failure.".format(filename))
            continue

        filename = os.path.basename(filename)
        filename_list.append(filename)
      

        
def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='datasets/', help='path of input, must be image directory')
    parser.add_argument('--dev_id', type=int, default=0, help='tpu id')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argsparser()
    main(args)
