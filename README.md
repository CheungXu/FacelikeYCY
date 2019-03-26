# FacelikeYCY
This Program can be used to calculate Face-Similarity with [YangChaoyue](https://weibo.com/u/5644764907).

[中文文档](https://github.com/CheungXu/FacelikeYCY/blob/master/README_CN.md)


## Environment
+ Ubuntu 18.04
+ Python 3.5
+ Tensorflow r1.12.0
+ OpenCV 4.0.0

## Usage

### 1、Clone Code & Download Model
（1）Clone Code:
```git
git clone https://github.com/CheungXu/FacelikeYCY
```
（2）Download model from [HERE](https://pan.baidu.com/s/1w0HFw4alVqpWTYJj5bO00Q)(Code:phfv).


（3）Unzip model file:
```bash
unzip model.zip
```

### 2、Run to Calculate Face-Similarity

```bash
python compare.py --src_path [first image path] --dst_path [second image path] --image_size [image size (Max 200)]
```

## Example & Result

Run:
```bash 
python compare.py --src_path ./data/ycy.jpg --dst_path ./data/ycy2.jpg --image_size 200
```

Result:
```bash
Face Distance: 0.74, Similarity Percent: 83.90
```

## Issue
If you find the bug and problem, Thanks for your issue to propose it.
 
## Reference code
[FACENET](https://github.com/davidsandberg/facenet)
