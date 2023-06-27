//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_H
#define TEST_H

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "bmnn_utils.h"
#include "engine.h"
#include "utils.hpp"
#include "bm_wrapper.hpp"
#include "cvwrapper.h"

// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#define DEBUG 0


class Test {
    std::shared_ptr<sail::Engine>              engine;
    std::shared_ptr<sail::Bmcv>                bmcv;
    std::vector<std::string>                   graph_names;    
    std::vector<std::string>                   input_names;    
    std::vector<int>                           input_shape;   //1 input
    std::vector<std::string>                   output_names;   
    std::vector<std::vector<int>>              output_shape;  //1 or 3 output  
    bm_data_type_t                             input_dtype;    
    bm_data_type_t                             output_dtype;   
    std::shared_ptr<sail::Tensor>              input_tensor;
    std::vector<std::shared_ptr<sail::Tensor>> output_tensor;
    std::map<std::string, sail::Tensor*>       input_tensors; 
    std::map<std::string, sail::Tensor*>       output_tensors; 

    cv::Mat stored_img;
    // configuration
    // 
    int m_net_h, m_net_w;
    int max_batch=1;
    int min_dim;
    float ab[6];
    int output_tensor_num;
    TimeStamp* m_ts;

   private:
    int pre_process(sail::BMImage& input);
    template <std::size_t N>
    int pre_process(std::vector<sail::BMImage>& input);
    int post_process(std::vector<sail::BMImage>& images);
    int argmax(float* data, int dsize);
    static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* alignWidth);
    

   public:
    Test(int dev_id, std::string bmodel_file);
    virtual ~Test();
    
    int Init();
    void enableProfile(TimeStamp* ts);
    int batch_size();
    int Detect(std::vector<sail::BMImage>& images);
    std::string palette;
};

#endif  //
