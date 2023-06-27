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


SELECT_NUMPY=False

class Test(object):
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSO)
        logging.debug("load {} success!".format(args.bmodel))
        # self.handle = self.net.get_handle()
        self.handle = sail.Handle(args.dev_id)
        self.bmcv = sail.Bmcv(self.handle)

        # load bmodel
        self.graph_name = self.net.get_graph_names()[0]
        # get input
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_dtype= self.net.get_input_dtype(self.graph_name, self.input_name)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.input_shapes = {self.input_name: self.input_shape}
        logging.debug(str(("graph_name: {}, input_names & input_shapes: ".format(self.graph_name), self.input_name, self.input_shapes)))
        
        # get output
        self.output_names = self.net.get_output_names(self.graph_name)
      
        self.output_name = self.output_names[0]
        self.output_scale = self.net.get_output_scale(self.graph_name, self.output_name)
        self.output_shape = self.net.get_output_shape(self.graph_name, self.output_name)
        self.output_dtype = self.net.get_output_dtype(self.graph_name, self.output_name)
        self.output_tensor = sail.Tensor(self.handle, self.output_shape, self.output_dtype,  True, True)
        self.output_tensors = {self.output_name: self.output_tensor}

        logging.debug(str(("graph_name: {}, output_names & output_shapes: ".format(self.graph_name), self.output_names, self.output_shape)))
        
   
        # 用于normalize
        #self.input_scale = float(1.0)

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        self.use_resize_padding = True
        self.use_vpp = False
        self.resize_img=sail.BMImage()

        # 修改图片
        # 模型输入大小
        self.net_sacle=(1024,512)
        self.flip=False
        self.flip_direction="horizontal"
        self.keep_ratio=True
        self.to_rgb=True

        # 归一化处理
        self.mean = [123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
        self.size_divisor=32
        # y=ax+b
        self.a = [1/x for x in self.std]
        self.b = [-x/y for x,y in zip(self.mean, self.std)]

        self.ab = []
        for i in range(3):
            self.ab.append(self.a[i]*self.input_scale)
            self.ab.append(self.b[i]*self.input_scale)

        # 时间计算
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time=0.0
    # a=self.show_img(rgb_planar_img)
    def show_img(self,img):
        self.bmcv.imwrite('result_pre1.png', img)
        pre_img=cv2.imread("result_pre1.png")
        return pre_img
    
    def preprocess_bmcv(self, input_bmimg):
        rgb_planar_img = sail.BMImage(self.handle, input_bmimg.height(), input_bmimg.width(),
                                          sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
        self.bmcv.convert_format(input_bmimg, rgb_planar_img)
        
        img_w = rgb_planar_img.width()
        img_h = rgb_planar_img.height()
        if self.use_resize_padding:
            r_w = self.net_w / img_w
            r_h = self.net_h / img_h
            if r_h > r_w:
                tw = self.net_w
                th = int(r_w * img_h)
                tx1 = tx2 = 0
                ty1 = int((self.net_h - th) / 2)
                ty2 = self.net_h - th - ty1
            else:
                tw = int(r_h * img_w)
                th = self.net_h
                tx1 = int((self.net_w - tw) / 2)
                tx2 = self.net_w - tw - tx1
                ty1 = ty2 = 0

            ratio = (min(r_w, r_h), min(r_w, r_h))
            txy = (tx1, ty1)
            attr = sail.PaddingAtrr()
            attr.set_stx(tx1)
            attr.set_sty(ty1)
            attr.set_w(tw)
            attr.set_h(th)
            attr.set_r(114)
            attr.set_g(114)
            attr.set_b(114)
            
            preprocess_fn = self.bmcv.vpp_crop_and_resize_padding if self.use_vpp else self.bmcv.crop_and_resize_padding
            resized_img_rgb = preprocess_fn(rgb_planar_img, 0, 0, img_w, img_h, self.net_w, self.net_h, attr)
        else:
            r_w = self.net_w / img_w
            r_h = self.net_h / img_h
            ratio = (r_w, r_h)
            txy = (0, 0)
            preprocess_fn = self.bmcv.vpp_resize if self.use_vpp else self.bmcv.resize
            resized_img_rgb = preprocess_fn(rgb_planar_img, self.net_w, self.net_h)

        # self.bmcv.imwrite("resize.png", resized_img_rgb);    
        # output_bmimg=self.ab[0]*resize_bmimg_rgb+self.ab[1]
        self.resize_img=resized_img_rgb

        output_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, \
                        sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
        self.bmcv.convert_to(resized_img_rgb, output_bmimg, \
                                       ((self.ab[0], self.ab[1]), \
                                       (self.ab[2], self.ab[3]), \
                                       (self.ab[4], self.ab[5])))
        return output_bmimg
        

    def predict(self, input_tensor):
        input_tensors = {self.input_name: input_tensor}
        self.net.process(self.graph_name, input_tensors, self.output_tensors)
        outputs = self.output_tensor.asnumpy() * self.output_scale
        return outputs[0]

    
   
    def postprocess(self,input_img, result):
        palette=pp.get_palette(args.palette)
        color_seg=pp.palette_bmcv(result,palette)
        if SELECT_NUMPY:
            #更换颜色通道
            # input_bmimg_bgr = self.bmcv.yuv2bgr(input_img)
            # resize_input_rgb = self.bmcv.resize(input_img, self.net_w,self.net_h)
            resize_input_rgb=self.bmcv.bm_image_to_tensor(input_img).asnumpy()
            # 删除第一个维度
            resize_input_rgb = np.squeeze(resize_input_rgb)
            # 更换形状使与color_seg一致
            resize_input_rgb=resize_input_rgb.transpose(1,2,0)
        
            output_bmimg=color_seg*0.5+resize_input_rgb*0.5
        else :
            # # 第二种方式k
            # color_seg.shape
            # (512, 1024, 3)
            # cv2.imwrite("color_seg.png", color_seg)
            # (3, 512, 1024)
            color_seg_transpose=color_seg.transpose(2,0,1)
            # 形状为：[1, 3, 512, 1024]
            color_seg_expand = np.expand_dims(color_seg_transpose, axis=0)
            
            color_seg_tensor=sail.Tensor(self.handle,color_seg_expand,False)
            color_seg_bmimg=self.bmcv.tensor_to_bm_image(color_seg_tensor)

            # self.bmcv.imwrite("color_seg_bmimg.png",color_seg_bmimg)  
            color_seg_rgb= sail.BMImage(self.handle, color_seg_bmimg.height(), color_seg_bmimg.width(),
                                          sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
            self.bmcv.convert_format(color_seg_bmimg, color_seg_rgb)
            # resize_color_seg_bmimg = self.bmcv.resize(color_seg_bmimg, input_img.he(), input_img.height())
            output_bmimg = sail.BMImage()
            # 图片叠加
            self.bmcv.image_add_weighted(color_seg_rgb, float(0.5), input_img, float(0.5), float(0.0),output_bmimg)
            # self.bmcv.imwrite('output_bmimg.png',output_bmimg)
        
        return output_bmimg

    def sav_result(self,palette_img):
        pp.save_and_show_palette_img(palette_img)


    def __call__(self, img_list):
        img_num = len(img_list)
        img_input_list = []
        
        for bmimg in img_list:
            output_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, \
                        sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
            start_time = time.time()
            output_bmimg = self.preprocess_bmcv(bmimg)
            self.preprocess_time += time.time() - start_time
            
            input_tensor = sail.Tensor(self.handle, self.input_shape,  self.input_dtype,  False, False)
            self.bmcv.bm_image_to_tensor(output_bmimg, input_tensor)
            start_time = time.time()
            outputs = self.predict(input_tensor)
            # print(np.shape(outputs))
            print(type(outputs))
            self.inference_time += time.time() - start_time
            
            start_time = time.time()
            res = self.postprocess(self.resize_img,outputs)
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
    test = Test(args)
    decode_time = 0.0

    filename_list = []
    results_list = []
    cn=0

    for filename in glob.glob(args.input+'/*'):
        if os.path.splitext(filename)[-1] not in ['.jpg','.png','.jpeg','.bmp','.JPEG','.JPG','.BMP']:
            continue
        start_time = time.time()
        decoder = sail.Decoder(filename, True, args.dev_id)
        bmimg = sail.BMImage()
        ret = decoder.read(test.handle, bmimg)
        if ret != 0:
            logging.error("{} decode failure.".format(filename))
            continue
        decode_time += time.time() - start_time

        filename = os.path.basename(filename)
        filename_list.append(filename)
        # inference 
        results = test([bmimg])
        results_list.append(results)
    
        # save image
        
        if SELECT_NUMPY:
            cv2.imwrite(os.path.join(output_img_dir, filename), results)
        else:
            handle = sail.Handle(args.dev_id)
            bmcv = sail.Bmcv(handle)
            bmcv.imwrite(os.path.join(output_img_dir, filename), results)


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
    parser.add_argument('--input', type=str, default='datasets/test/', help='path of input, must be image directory')
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
