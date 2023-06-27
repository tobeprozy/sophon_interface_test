#!/bin/bash
script_dir=$(dirname $(readlink -f "$0"))


# 1684X yes 1684 ing
if [ ! $1 ];then
    echo "Please set the target chip. Option: BM1684 and BM1684X"
    exit
else
    target=$1
fi

outdir=../models/$target
cali_data_path=../datasets/cali_data

function auto_cali(){
    python3 -m ufw.cali.cali_model \
        --net_name=segformer \
        --model=../models/onnx/segformer.b0.512x1024.city.160k.onnx \
        --cali_image_path=../datasets/cali_data4 \
        --cali_image_preprocess='resize_h=512,resize_w=1024' \
        --input_shape '[1,3,512,1024]' \
        --target=$target
       
    if [ $? -ne 0 ]; then
        echo  "gen_int8bmodel failed"
    else
        mv mv ../models/onnx/segformer_batch1/compilation.bmodel $outdir/segformer_int8.b0.512x1024.city.160k.bmodel
    fi
}
pushd $script_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
auto_cali

popd