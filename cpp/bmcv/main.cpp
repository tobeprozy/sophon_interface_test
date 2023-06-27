//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <fstream>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include "json.hpp"
#include "opencv2/opencv.hpp"
#include "ff_decode.hpp"
#include "test.hpp"
using json = nlohmann::json;
using namespace std;


using namespace std;

int main(int argc, char *argv[])
{
  cout.setf(ios::fixed);
  cout << "===========================+++++++++++++++++++++++++++" << endl;

  int dev_id = 0;

  // creat handle
  BMNNHandlePtr handle = make_shared<BMNNHandle>(dev_id);
  cout << "set device id: " << dev_id << endl;
  bm_handle_t h = handle->handle();


  string img_file="../../datasets/demo.png";

  bm_image bmimg;
  picDec(h, img_file.c_str(), bmimg);

  size_t index = img_file.rfind("/");
  string img_name = img_file.substr(index + 1);

    
  return 0;
}
