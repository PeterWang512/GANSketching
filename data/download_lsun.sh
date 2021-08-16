#!/bin/bash
mkdir -p image/lmdb
wget http://dl.yf.io/lsun/objects/cat.zip -O ./data/image/cat.zip
wget http://dl.yf.io/lsun/objects/horse.zip -O ./data/image/horse.zip
wget http://dl.yf.io/lsun/scenes/church_outdoor_train_lmdb.zip -O ./data/image/church.zip
unzip ./data/image/cat.zip -d ./data/image/lmdb
unzip ./data/image/horse.zip -d ./data/image/lmdb
unzip ./data/image/church.zip -d ./data/image/lmdb
python ./data/prepare_lsun.py ./data/image/cat ./data/image/lmdb/cat
python ./data/prepare_lsun.py ./data/image/horse ./data/image/lmdb/horse
python ./data/prepare_lsun.py ./data/image/church ./data/image/lmdb/church_outdoor_train_lmdb
rm -r ./data/image/lmdb
rm ./data/image/cat.zip
rm ./data/image/horse.zip
rm ./data/image/church.zip
