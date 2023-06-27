#!/bin/bash
# 将当前脚本文件的目录路径赋值给变量 model_dir
model_dir=$(dirname $(readlink -f "$0"))

# 可以生成BM1684X, NO 1684

#默认为bm1684x
if [ ! $1 ]; 
then
    target=BM1684X
else
    target=$1
fi

outdir=../models/$target

function gen_mlir(){
    model_transform.py \
        --model_name segformer \
        --model_def ../models/onnx/segformer.b0.512x1024.city.160k.onnx \
        --input_shapes [[$1,3,512,1024]]  \
        ----keep_aspect_ratio \
	--mean 123.675,116.28,103.53 \
        --mlir segformer.mlir 

}

function gen_fp32bmodel(){
    model_deploy.py \
        --mlir segformer.mlir \
        --quantize F32 \
        --chip $target \
        --model segformer_fp32.b0.512x1024.city.160k.bmodel
    mv segformer_fp32.b0.512x1024.city.160k.bmodel $outdir/

}

# 从当前目录进入modol_dir
pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

# batch_size=1
gen_mlir 1
gen_fp32bmodel 1

# 回到当前目录
popd
