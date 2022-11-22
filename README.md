## 1. 安装依赖包

```bash
pip install -r requirements.txt
```

## 2. 命令行运行

```bash
streamlit run feedback_deploy.py
```

因为最终版本其实只有feedback_deploy.py所以只要运行它即可。





**文件说明**（主文件为feedback_deploy.py）：

1. yolov5-master 官方源文件用于导入模型

2. feedback文件夹用于存储反馈信息

3. 123.mp4视频预测的结果视频

4. best.pt算式识别权重文件

5. openh264-1.8.0-win64.dll用于解决opencv的一些问题

6. voc2yolo.py voc格式转yolo格式

7. pic_deploy.py 图片识别推理部署代码

8. video_deploy.py 在7基础上添加视频

9. yolo_deploy.py 在8的基础上添加算式识别模块

10. feedback_deploy.py在9的基础上添加反馈模块