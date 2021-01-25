#darkutils

## build
Build & install darknet and darkhelp first:
```
# build & install darknet
sudo apt-get install build-essential git libopencv-dev

cd ~/
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
vi Makefile # to set LIBSO=1, and possibly other flags
make
sudo cp libdarknet.so /usr/local/lib/
sudo cp include/darknet.h /usr/local/include/
sudo ldconfig

# build && install darkhelp
sudo apt-get install cmake libtclap-dev libmagic-dev libopencv-dev
cd ~
git clone https://github.com/Roman-/darkhelp.git
cd darkhelp && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
make package
sudo dpkg -i ./darkhelp-1.1.3-3101-Linux-x86_64-Ubuntu-20.04.deb
```
