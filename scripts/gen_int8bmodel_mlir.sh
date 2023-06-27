#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
# 可以生成BM1684X, NO 1684
if [ ! $1 ]; then
    target=BM1684X
else
    target=$1
fi

outdir=../models/$target

function gen_mlir()
{
   model_transform.py \
        --model_name segformer \
        --model_def ../models/onnx/segformer.b0.512x1024.city.160k.onnx \
        --input_shapes [[1,3,512,1024]] \
        --keep_aspect_ratio \
        --mean 123.675,116.28,103.53 \
        --mlir segformer.mlir
}

function gen_cali_table()
{
    run_calibration.py segformer.mlir \
        --dataset ../datasets/cali_data4/ \
        --input_num 100 \
        -o segformer_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir segformer.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table segformer_cali_table \
        --model segformer_int8.b0.512x1024.city.160k.bmodel

    mv segformer_int8.b0.512x1024.city.160k.bmodel $outdir/
}


pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir
gen_cali_table
gen_int8bmodel

# batch_size=4
# gen_mlir 4
# gen_int8bmodel 4

popd