//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "test.hpp"
#include <fstream>
#include <vector>
#include <string>
#define USE_ASPECT_RATIO 1 
#define DUMP_FILE 0


Test::Test(std::shared_ptr<BMNNContext> context) : m_bmContext(context)
{
  std::cout << "Test construct success !!!" << std::endl;
}

Test::~Test()
{
  std::cout << "Test deconstruct ing" << std::endl;
  bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
  bm_image_free_contiguous_mem(max_batch, m_converto_imgs.data());
  for (int i = 0; i < max_batch; i++)
  {
    bm_image_destroy(m_converto_imgs[i]);
    bm_image_destroy(m_resized_imgs[i]);
  }
}

int Test::Init()
{
  // 1. get network
  m_bmNetwork = m_bmContext->network(0);
  // for debug
  auto out_tensor = m_bmNetwork->outputTensor(0);
  // 2. get input
  max_batch = m_bmNetwork->maxBatch();
  auto tensor = m_bmNetwork->inputTensor(0);
  m_net_h = tensor->get_shape()->dims[2];
  m_net_w = tensor->get_shape()->dims[3];

  // 3. get output
  output_num = m_bmNetwork->outputTensorNum();
  assert(output_num > 0);

  // 4. initialize bmimages
  m_resized_imgs.resize(max_batch);
  m_converto_imgs.resize(max_batch);
  // some API only accept bm_image whose stride is aligned to 64
  int aligned_net_w = FFALIGN(m_net_w, 64);
  int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
  for (int i = 0; i < max_batch; i++)
  {
    auto ret = bm_image_create(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &m_resized_imgs[i], strides);
    assert(BM_SUCCESS == ret);
  }
  bm_image_alloc_contiguous_mem(max_batch, m_resized_imgs.data());
  bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
  if (tensor->get_dtype() == BM_INT8)
  {
    img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
  }
  auto ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, img_dtype, m_converto_imgs.data(), max_batch);
  assert(BM_SUCCESS == ret);

  // 5.converto
  float input_scale = tensor->get_scale();
  const std::vector<float> std = {58.395, 57.12, 57.375};
  const std::vector<float> mean = {123.675, 116.28, 103.53};
  converto_attr.alpha_0 = 1 / (std[0]) * input_scale;
  converto_attr.alpha_1 = 1 / (std[1]) * input_scale;
  converto_attr.alpha_2 = 1 / (std[2]) * input_scale;
  converto_attr.beta_0 = (-mean[0] / std[0]) * input_scale;
  converto_attr.beta_1 = (-mean[1] / std[1]) * input_scale;
  converto_attr.beta_2 = (-mean[2] / std[2]) * input_scale;

  return 0;
}

void Test::enableProfile(TimeStamp *ts)
{
  m_ts = ts;
}

int Test::batch_size()
{
  return max_batch;
};

int Test::Detect(std::vector<bm_image> &input_images)
{
  int ret = 0;
  // 3. preprocess
  LOG_TS(m_ts, "Test preprocess");
  ret = pre_process(input_images);
  CV_Assert(ret == 0);
  LOG_TS(m_ts, "Test preprocess");

  // 4. forward
  LOG_TS(m_ts, "Test inference");
  ret = m_bmNetwork->forward();
  CV_Assert(ret == 0);
  LOG_TS(m_ts, "Test inference");

  // 5. post process
  LOG_TS(m_ts, "Test postprocess");
  ret = post_process(input_images);
  CV_Assert(ret == 0);
  LOG_TS(m_ts, "Test postprocess");
  return ret;
}

int Test::pre_process(const std::vector<bm_image> &images)
{
  std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
  int image_n = images.size();

  // 1.1 resize image
  int ret = 0;
  for (int i = 0; i < image_n; ++i)
  {
    bm_image image1 = images[i];
    int w = image1.width;
    bm_image image_aligned;
    bool need_copy = image1.width & (64 - 1);
    if (need_copy)
    {
      int stride1[3], stride2[3];
      bm_image_get_stride(image1, stride1);
      stride2[0] = FFALIGN(stride1[0], 64);
      stride2[1] = FFALIGN(stride1[1], 64);
      stride2[2] = FFALIGN(stride1[2], 64);
      bm_image_create(m_bmContext->handle(), image1.height, image1.width,
                      image1.image_format, image1.data_type, &image_aligned, stride2);

      bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN);

      bmcv_copy_to_atrr_t copyToAttr;
      memset(&copyToAttr, 0, sizeof(copyToAttr));
      copyToAttr.start_x = 0;
      copyToAttr.start_y = 0;
      copyToAttr.if_padding = 1;
      bmcv_image_copy_to(m_bmContext->handle(), copyToAttr, image1, image_aligned);
    }
    else
    {
      image_aligned = image1;
    }

#if 1
    bool isAlignWidth = false;
    float ratio = get_aspect_scaled_ratio(images[i].width, images[i].height, m_net_w, m_net_h, &isAlignWidth);
    bmcv_padding_atrr_t padding_attr;
    memset(&padding_attr, 0, sizeof(padding_attr));
    padding_attr.dst_crop_sty = 0;
    padding_attr.dst_crop_stx = 0;
    padding_attr.padding_b = 114;
    padding_attr.padding_g = 114;
    padding_attr.padding_r = 114;
    padding_attr.if_memset = 1;
    if (isAlignWidth)
    {
      padding_attr.dst_crop_h = images[i].height * ratio;
      padding_attr.dst_crop_w = m_net_w;

      int ty1 = (int)((m_net_h - padding_attr.dst_crop_h) / 2);
      padding_attr.dst_crop_sty = ty1;
      padding_attr.dst_crop_stx = 0;
    }
    else
    {
      padding_attr.dst_crop_h = m_net_h;
      padding_attr.dst_crop_w = images[i].width * ratio;

      int tx1 = (int)((m_net_w - padding_attr.dst_crop_w) / 2);
      padding_attr.dst_crop_sty = 0;
      padding_attr.dst_crop_stx = tx1;
    }

    bmcv_rect_t crop_rect{0, 0, image1.width, image1.height};
    auto ret = bmcv_image_vpp_convert_padding(m_bmContext->handle(), 1, image_aligned, &m_resized_imgs[i],
                                              &padding_attr, &crop_rect);
#else

    bm_image image2 = images[i];
    bm_image_write_to_bmp(images[i],"images.bmp");
    bmcv_rect_t crop_rect{740, 364, 212, 179};
    auto ret = bmcv_image_vpp_convert(m_bmContext->handle(), 1, images[i], &image2,&crop_rect);
    bm_image_write_to_bmp(image2,"image2.bmp");
    
    bm_image image3 = images[i];
    bm_image image4 = images[i];   
    int crop_num_vec = 1;
    bmcv_image_vpp_basic(m_bmContext->handle(), 1,&image4, &image3, &crop_num_vec, &crop_rect, NULL);
    bm_image_write_to_bmp(image3,"image3.bmp");

#endif
    assert(BM_SUCCESS == ret);

#if DUMP_FILE
    cv::Mat resized_img;
    cv::bmcv::toMAT(&m_resized_imgs[i], resized_img);
    std::string fname = cv::format("resized_img_%d.jpg", i);
    cv::imwrite(fname, resized_img);
#endif
    // bm_image_destroy(image1);
    if (need_copy)
      bm_image_destroy(image_aligned);
  }

  // 1.2 converto
  ret = bmcv_image_convert_to(m_bmContext->handle(), image_n, converto_attr, m_resized_imgs.data(), m_converto_imgs.data());
  CV_Assert(ret == 0);

  // 1.3 attach to tensor
  if (image_n != max_batch)
    image_n = m_bmNetwork->get_nearest_batch(image_n);
  bm_device_mem_t input_dev_mem;

  
  bm_image_get_contiguous_device_mem(image_n, m_converto_imgs.data(), &input_dev_mem);
  input_tensor->set_device_mem(&input_dev_mem);
  input_tensor->set_shape_by_dim(0, image_n); // set real batch number
  return 0;
}

float Test::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *pIsAligWidth)
{
  float ratio;
  float r_w = (float)dst_w / src_w;
  float r_h = (float)dst_h / src_h;
  if (r_h > r_w)
  {
    *pIsAligWidth = true;
    ratio = r_w;
  }
  else
  {
    *pIsAligWidth = false;
    ratio = r_h;
  }
  return ratio;
}

int Test::post_process(std::vector<bm_image> &images)
{
  std::vector<std::shared_ptr<BMNNTensor>> outputTensors(output_num);
  for (int i = 0; i < output_num; i++)
  {
    outputTensors[i] = m_bmNetwork->outputTensor(i);
  }

  for (int batch_idx = 0; batch_idx < images.size(); ++batch_idx)
  {
    auto output_shape = outputTensors[batch_idx]->get_shape();
    auto output_dims = output_shape->num_dims; //[1 1 512 1024]

    auto output_data = outputTensors[batch_idx]->get_cpu_data();

#if DEBUG
    std::cout << "Output shape infos: " << output_shape->dims[0] << " "
              << output_shape->dims[1] << " " << output_shape->dims[2] << " "
              << output_shape->dims[3] << std::endl;
#endif

    std::cout << "get output color_seg ...." << std::endl;

    // 1. Get output bounding boxes.
    int raw_height = output_shape->dims[0] *
                     output_shape->dims[1] *
                     output_shape->dims[2];
    int col_width = output_shape->dims[3];

    std::vector<std::vector<float>> output_array(raw_height, std::vector<float>(col_width, 0));
    for (int row = 0; row < raw_height; row++)
    {
      for (int col = 0; col < col_width; col++)
      {
        output_array[row][col] = *(output_data++);
      }
    }

    

  
  }

  return 0;
}
