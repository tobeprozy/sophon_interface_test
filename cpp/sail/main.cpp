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
#define USE_OPENCV_DECODE 0

using namespace std;

int main(int argc, char *argv[])
{
    cout.setf(ios::fixed);
    cout << "===========================+++++++++++++++++++++++++++" << endl;

    int dev_id = 1;

    // create handle
    auto handle = sail::Handle(dev_id);
    sail::Bmcv bmcv(handle);
    cout << "set device id: " << dev_id << endl;


    sail::BMImage batch_img;
    string img_file="../../datasets/demo.png";
    sail::Decoder decoder((const string)img_file, true, dev_id);
    int ret = decoder.read(handle, batch_img);
    if (ret != 0)
    {
        cout << "read failed"
             << "\n";
    }

    return 0;
}