#!/bin/bash

#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))


pushd $model_dir

echo "1"
# 执行./gen_fp32bmodel_mlir.sh BM1684X
./gen_fp32bmodel_mlir.sh BM1684X
echo "2"
# 执行./gen_fp32bmodel_mlir.sh BM1684
./gen_fp32bmodel_mlir.sh BM1684
echo "3"
# 执行./gen_fp16bmodel_mlir.sh BM1684X
./gen_fp16bmodel_mlir.sh BM1684X
echo "4"
# 执行./gen_int8bmodel_mlir.sh BM1684X
./gen_int8bmodel_mlir.sh BM1684X
echo "5"
# 执行./gen_int8bmodel_mlir.sh BM1684
./gen_int8bmodel_mlir.sh BM1684

popd
