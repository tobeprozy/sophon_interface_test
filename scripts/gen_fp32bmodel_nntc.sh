#!/bin/bash

# BM1684和X都可以
# 将当前脚本文件的目录路径赋值给变量 model_dir
model_dir=$(dirname $(readlink -f "$0"))

#默认为bm1684x
if [ ! $1 ]; 
then
    echo "Please set the target chip. Option: BM1684 and BM1684X"
    exit
else
    target=$1
fi

outdir=../models/$target
# 这里的$1和上面的不一样，上面的是脚本的参数，下面是函数的参数
function gen_fp32bmodel(){
    python3 -m bmneto \
        --model ../models/onnx/segformer.b0.512x512.ade.160k.onnx \
        --target $target \
        --shapes [[1,3,256,256]]  \
        --opt 2 \
        --dyn False
    mv compilation/compilation.bmodel $outdir/segformer_fp32_b0_ade.bmodel
}

pushd $model_dir
pwd
if [ ! -d $outdir ];then
    mkdir -p $outdir
fi
#batch_size=1
gen_fp32bmodel 4

popd