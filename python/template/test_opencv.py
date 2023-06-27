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

RESEZE_ORIGIN=0


class Test(object):
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.debug("load {} success!".format(args.bmodel))

        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name)
        self.input_shapes = [self.net.get_input_shape(self.graph_name, name) for name in self.input_names]
        logging.debug(str(("graph_name: {}, input_names & input_shapes: ".format(self.graph_name), self.input_names, self.input_shapes)))
        
        self.output_names = self.net.get_output_names(self.graph_name)
        self.output_shapes = [self.net.get_output_shape(self.graph_name, name) for name in self.output_names]
        logging.debug(str(("graph_name: {}, output_names & output_shapes: ".format(self.graph_name), self.output_names, self.output_shapes)))
        
        
        self.input_name = self.input_names[0]
        self.input_shape = self.input_shapes[0]

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        # 修改图片
        # 模型输入大小
        self.net_sacle=(self.net_w, self.net_h)
        self.flip=False
        self.flip_direction="horizontal"
        self.keep_ratio=True
        self.to_rgb=True

        self.post_img_input=[]
        # 归一化处理
        self.mean = [123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
        self.size_divisor=32

        # 时间计算
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    # align 
    def _align(self, img, size_divisor, interpolation=None):
        align_h = int(np.ceil(img.shape[0] / size_divisor)) * size_divisor
        align_w = int(np.ceil(img.shape[1] / size_divisor)) * size_divisor
        if interpolation == None:
            img = cv2.resize(img, (align_w, align_h))
        else:
            img = cv2.resize(img, (align_w, align_h), interpolation=interpolation)
        return img
    

    def _resize_img(self, results):
        img = results['img']
        # self.keep_ratio=False
        if self.keep_ratio:
            shape = img.shape[:2]
             # Scale ratio (new / old)
            r = min(self.net_sacle[0] / shape[1], self.net_sacle[1] / shape[0])
            # Compute padding
            ratio = r, r  # width, height ratios
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            dh, dw = self.net_sacle[1] - new_unpad[1], self.net_sacle[0] - new_unpad[0]  # wh padding

            dw /= 2  # divide padding into 2 sides
            dh /= 2
            if shape[::-1] != new_unpad:  # resize
                img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
            # cv2.imwrite("img.png",img);
        else:
            # 使用cv2.resize函数进行图像缩放
            img = cv2.resize(img,self.net_sacle)
          
    
        results['img']=img

        return img 
    
    def _flip(self,results):
        if 'flip' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
        # flip image
            results['img'] = cv2.flip(results['img'], 1 if results['flip_direction'] == 'horizontal' else 0)
            # flip segs
            for key in results.get('seg_fields', []):
                # use copy() to make numpy stride positive
                results[key] = cv2.flip(results[key], 1 if results['flip_direction'] == 'horizontal' else 0).copy()
        return results
    
    def _normalize(self,results,to_rgb=True):
        self.mean = np.array(self.mean, dtype=np.float32)
        self.std = np.array(self.std, dtype=np.float32)
        self.to_rgb = to_rgb
    
        img=results['img']
        img = img.copy().astype(np.float32)

        assert img.dtype != np.uint8
        mean = np.float64(self.mean.reshape(1, -1))
        stdinv = 1 / np.float64(self.std.reshape(1, -1))
        if to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        results['img']=img
    
    def _totensor(self, results):
        img = results["img"]
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        results["img"] = img.transpose(2, 0, 1) 
        return results
    
    def preprocess(self, img):
        self.origin_scale=img.shape 
        results={}
        # 图像大小调整
        results['img'] = img
        results['flip']= self.flip
        results['flip_direction']=self.flip_direction
        results['keep_ratio']=self.keep_ratio
        results['to_rgb']=self.to_rgb

        """Resize images with ``results['scale']``."""
        img=self._resize_img(results)
        
        self.post_img_input.append(img)
        """Call function to flip bounding boxes, masks, semantic segmentation maps"""
        self._flip(results)
        self._normalize(results)
        self._totensor(results)
        # 返回处理后的数据
        return results['img']


    def predict(self, input_img):
        input_data = {self.input_name: input_img}
        outputs = self.net.process(self.graph_name, input_data)
        return [list(outputs.values())[0][0][0]]

    def postprocess(self,input_img, result):
        palette=pp.get_palette(args.palette)
        img=pp.palette_img(input_img,result,palette)
        if RESEZE_ORIGIN:
        # 还原原来的图片大小
            if self.origin_scale!=img.shape:
                h, w=self.origin_scale[0],self.origin_scale[1]
                img=cv2.resize(img,(w,h))
        return img

    def sav_result(self,palette_img):
        pp.save_and_show_palette_img(palette_img)


    def __call__(self, img_list):
        img_num = len(img_list)
        img_input_list = []
        for img in img_list:
            start_time = time.time()
            img = self.preprocess(img)
            self.preprocess_time += time.time() - start_time
            img_input_list.append(img)
        
        if img_num == self.batch_size:
            input_img = np.stack(img_input_list)
            start_time = time.time()
            outputs = self.predict(input_img) 
            self.inference_time += time.time() - start_time
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(img_input_list)
            start_time = time.time()
            outputs = self.predict(input_img)[:img_num]
            self.inference_time += time.time() - start_time
        
        for img in self.post_img_input:
            start_time = time.time()
            res = self.postprocess(img,outputs)
            self.postprocess_time += time.time() - start_time
        return res

    def get_time(self):
        return self.dt

def main(args):
    
    # creat save path
    output_dir = "python/results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir) 
    

    # initialize net
    Test = Test(args)
    decode_time = 0.0

    filename_list = []
    results_list = []
    cn=0

    for filename in glob.glob(args.input+'/*'):
        if os.path.splitext(filename)[-1] not in ['.jpg','.png','.jpeg','.bmp','.JPEG','.JPG','.BMP']:
            continue

        start_time = time.time()
        src_img = cv2.imread(filename)
        print("open :"+filename)
        if src_img is None:
            logging.error("{} imread is None.".format(filename))
            continue
        decode_time += time.time() - start_time

        filename = os.path.basename(filename)
        filename_list.append(filename)
        # inference 
        results = Test([src_img])
        results_list.append(results)

        # save image

        cv2.imwrite(os.path.join(output_img_dir, filename), results)
        # 显示图片
        # pp.save_and_show_palette_img(results,out_file="demo1.png",show=True)
        # pp.save_and_show_palette_img(results,show=False,out_file="out_img.png")}
            
    # calculate speed  
    cn = len(results_list)    
    logging.info("------------------ Inference Time Info ----------------------")
    decode_time = decode_time / cn
    preprocess_time = Test.preprocess_time / cn
    inference_time = Test.inference_time / cn
    postprocess_time = Test.postprocess_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))
        
def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='datasets/test', help='path of input, must be image directory')
    parser.add_argument('--bmodel', type=str, default='models/BM1684/Test_fp32.b0.512x1024.city.160k.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='tpu id')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argsparser()
    main(args)
