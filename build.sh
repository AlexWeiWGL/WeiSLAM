#!/bin/bash

echo "Configuring and building Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW2 || return
mkdir build
cd build || return
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

cd ../../g2o || return
echo "Configuring and building Thirdparty/g2o ..."

mkdir build
cd build || return
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

cd ../../../

echo "Uncompress vocabulary ..."

cd Vocabulary || return
tar -xf ORBvoc.txt.tar.gz
cd .. || return

echo "Configuring and building WeiSLAM ..."

mkdir build
cd build || return
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4