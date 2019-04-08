# FacelikeYCY
本项目用于计算与 [杨超越](https://weibo.com/u/5644764907)的人脸相似度，同时也可用于计算任意两人脸的相似度计算。本项目中，无需单独检测对齐人脸图像，程序会自动从照片中检测人脸进行对比。

## 代码运行环境：
+ Ubuntu 18.04
+ Python 3.5
+ Tensorflow r1.12.0
+ Opencv 4.0.0

## 本地使用

### 1、克隆代码并下载预训练模型
（1）克隆代码:
```git
git clone https://github.com/CheungXu/FacelikeYCY
```
（2）从[HERE](https://pan.baidu.com/s/1w0HFw4alVqpWTYJj5bO00Q)(提取码:phfv)下载预训练模型。


（3）解压模型文件:
```bash
unzip model.zip
```

### 2、执行相似度计算脚本

```bash
python compare.py --src_path [第一张图片路径] --dst_path [第二张图片路径] --image_size [计算相似度时图像大小（最大200）]
```

### 3、示例及结果

运行:
```bash 
python compare.py --src_path ./data/ycy.jpg --dst_path ./data/ycy2.jpg --image_size 200
```

结果:
```bash
Face Distance: 0.74, Similarity Percent: 83.90
```
## 服务器&客户端模式
### 1、Start Service

运行:
```bash 
python server.py
```
输出：
```bash
 * Serving Flask app "server" (lazy loading)
 * Environment: production
   WARNING: Do not use the development server in a production environment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://0.0.0.0:6006/ (Press CTRL+C to quit)
```
服务默认监听本机的6006端口。

### 2、请求并获取结果

运行:
```bash 
python client.py
```

结果：
```bash
<Response [200]>
{"score":"84.84680122799344"}
Time: 0.188511s
```
查看代码获取更多详细信息。

## Issue
如果有任何问题，欢迎提交Issue。
 
## 参考代码
[FACENET](https://github.com/davidsandberg/facenet)
