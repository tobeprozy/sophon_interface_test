#!/bin/bash
pip3 install dfn
sudo apt install unzip

script_dir=$(dirname $(readlink -f "$0"))
echo $script_dir

pushd $script_dir

# datasets
if [ ! -d "../datasets" ];
then
    mkdir -p ../datasets
    # test dataset
    # test input_test
    python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/bUIGQIdHo
    unzip input_test.zip -d ../datasets/
    rm input_test.zip
    echo "input_test download!"
else
    echo "datasets exist!"
fi


# models
if [ ! -d "../models" ]; 
then
    python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/UNCekgYxP
    ls -al
    unzip models.zip -d ../
    rm models.zip
    echo "models download!"
else
    echo "models exist!"
fi
popd