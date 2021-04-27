# This project is currently work-in-progress

## build
Build & install darknet and darkhelp first:

```bash
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

## Fixing your darknet dataset
When you execute
```bash
./darkutils validate ../data/tests/masks_cfg_weights/yolov4-tiny-masks2.cfg ../data/tests/masks_cfg_weights/yolov4-tiny-masks2.we
ights ../data/tests/masks_files/obj.names ../data/tests/masks_files/ result.duv.tsv
```
the result.duv.tsv file will be generated. Each line of the file has the following format (tab-separated):
```
filename c x y w h p iou
```
, where `c` is integer class id starting with 0, `x,y,w,h` are relative coords of detection (0-1), `p` is darknet probability (0-1), iou is IntersectionOverUnion (0-1) between your mark bbox and what darknet has predicted, filename is name of image file without extension and it may contain spaces.

# how to interpret .duv.tsv results

Let probThresh = 0.15, iouThresh = 0.45. Then:

- p > probThresh, iou > iouThresh means everything is allright with this mark.
- p > probThresh, iou < iouThresh means darknet has detected something that you haven't marked. Either you missed a mark OR darknet mistakenly spotted a thing. **The greater the `p` value, the more likely you have missed the mark**.
- p = 0, iou = 0 means darknet doesn't see what you've marked. Either you've marked it by mistake or you haven't trained darknet good enough yet.
