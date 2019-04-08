# FacelikeYCY
This Program can be used to calculate Face-Similarity with [YangChaoyue](https://weibo.com/u/5644764907).

[中文文档](https://github.com/CheungXu/FacelikeYCY/blob/master/README_CN.md)


## Environment
+ Ubuntu 18.04
+ Python 3.5
+ Tensorflow r1.12.0
+ OpenCV 4.0.0

## Local Usage 

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

### 3、Example & Result

Run:
```bash 
python compare.py --src_path ./data/ycy.jpg --dst_path ./data/ycy2.jpg --image_size 200
```

Result:
```bash
Face Distance: 0.74, Similarity Percent: 83.90
```

## Server&Client Mode

### 1、Start Service

Run:
```bash 
python server.py
```
Output：
```bash
 * Serving Flask app "server" (lazy loading)
 * Environment: production
   WARNING: Do not use the development server in a production environment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://0.0.0.0:6006/ (Press CTRL+C to quit)
```
The service is listening on port 6006 by default.

### 2、Post and Get Result
Run:
```bash 
python client.py
```

Resutl：
```bash
<Response [200]>
{"score":"84.84680122799344"}
Time: 0.188511s
```
See the code for more information.

## Issue
If you find the bug and problem, Thanks for your issue to propose it.
 
## Reference code
[FACENET](https://github.com/davidsandberg/facenet)
