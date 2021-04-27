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
git clone https://github.com/stephanecharette/DarkHelp.git
cd DarkHelp && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
make package
sudo dpkg -i ./*.deb
```

## Fixing your darknet dataset
When you execute
```bash
./darkutils validate ../data/tests/masks_cfg_weights/yolov4-tiny-masks2.cfg ../data/tests/masks_cfg_weights/yolov4-tiny-masks2.we
ights ../data/tests/masks_files/obj.names ../data/tests/masks_files/ result.duv.tsv
```
the result.duv.tsv file will be generated. Each line of the file has the following format (tab-separated):
```
path c x y w h p iou treated
```
, where `path` is path to image file without extension, e.g. masks2/img0001. It's either relative to the location of train.txt file or an absolute path, depending on how the images are referred in train.txt. May contain whitespaces but not tabs.

`c` is integer class id starting with 0, `x,y,w,h` are relative coords of detection (0-1), `p` is darknet probability (0-1), iou is IntersectionOverUnion (0-1) between your mark bbox and what darknet has predicted.

The last value `treated` is single char 't' or 'f' which is used when you **cure** your dataset. By default they're all 'f' which stays for false. As you view the dataset and add/skip your potentially erroneous marks, viewed detections becomes maked as 't'. When this happens, original file.duv.tsv is overwritten.

# how to interpret .duv.tsv results

Let probThresh = 0.15, iouThresh = 0.45. Then:

- p > probThresh, iou > iouThresh means everything is allright with this mark.
- p > probThresh, iou < iouThresh means darknet has detected something that you haven't marked. Either you missed a mark OR darknet mistakenly spotted a thing. **The greater the `p` value, the more likely you have missed the mark**.
- p = 0, iou = 0 means darknet doesn't see what you've marked. Either you've marked it by mistake or you haven't trained darknet good enough yet.
