# yolo_slowfast

still updating, not completing yet

### introduction

We imply yolov11 and deepsort_oh for object detection and object tracking (based on the former project writer did linked here), together with SlowFast model (official) for action detection.

### how to run

my GPU Cuda version is 12.4

My python version is 3.12

we use the official version of SlowFast, so we have to make the environment of SlowFast first, but follow  the following instructions is OK.

first we download the official SlowFast from (Thanks to reference No.2)

```
git clone https://gitee.com/qiang_sun/SlowFast.git
```

then

```
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/fvcore'
conda install av==14.2.0 -c conda-forge
pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
```

then

```
export PYTHONPATH=/path/to/SlowFast/slowfast:$PYTHONPATH
```

Here u should change the "/path/to/" part to your own route

then run

```
python run.py --cfg SlowFast/demo/AVA/SLOWFAST_32x2_R101_50_50.yaml
```

Let's START!

### Reference

SlowFast：https://github.com/facebookresearch/SlowFast/tree/main

SlowFast的辛酸复现过程：https://blog.csdn.net/normal_lk/article/details/126138119
