# 机器人视觉控制攻击防御系统

这是一个基于PyQt5的机器人视觉控制攻击防御系统。系统提供了实时摄像头画面捕获、物体识别、对抗样本生成以及多种防御方法的功能。

## 功能特点

1. **识别页面**
   - 实时摄像头画面显示
   - 图像捕获功能
   - 物体坐标识别

2. **欺骗页面**
   - 显示原始捕获图像
   - 生成对抗样本
   - 对抗样本识别

3. **防御页面**
   - 多种防御算法支持
   - 防御效果对比
   - 防御后的识别结果显示

## 安装要求

- Python 3.8+
- OpenCV
- PyQt5
- Flask
- NumPy
- Pillow
- PyTorch
- Torchvision

## 安装步骤

1. 克隆项目到本地：
```bash
git clone [项目地址]
cd [项目目录]
```

2. 创建虚拟环境（推荐）：
```bash
# python version: 3.11.9
python -m venv .venv
```

3. 激活虚拟环境：
- Windows:
```bash
.venv\Scripts\activate
```
- Linux/Mac:
```bash
source .venv/bin/activate
```

4. 安装依赖：
```bash
pip install -r requirements.txt
```

## 运行方法

1. 启动PyQt5前端界面：
```bash
python main.py
```

## 使用说明

1. **识别功能**
   - 打开程序后，默认进入识别页面
   - 左侧显示实时摄像头画面
   - 点击"捕获"按钮可以截取当前画面
   - 点击"开始识别"按钮进行物体识别

2. **欺骗功能**
   - 切换到欺骗页面
   - 显示之前捕获的图像
   - 点击"生成对抗样本"生成对抗样本
   - 点击"开始识别"查看对抗效果

3. **防御功能**
   - 切换到防御页面
   - 显示生成的对抗样本
   - 选择不同的防御算法进行处理
   - 点击各个防御结果下的"开始识别"按钮查看防御效果

## 注意事项

1. 确保系统已安装摄像头并能正常使用
2. 运行程序时需要同时启动后端服务和前端界面
3. 默认后端服务端口为5000，请确保该端口未被占用
4. 运行main.py无法打开摄像头的话，有可能缺失VC依赖，运行VC_redist.x64.exe安装VC依赖包

## 开发说明

- 前端使用PyQt5开发，提供友好的图形界面
- 图像处理使用OpenCV和NumPy
- 深度学习相关功能使用PyTorch 

## 其他代码说明

- main.py: 主程序入口，启动PyQt5界面
- pages/: 包含各个功能页面的代码
- face_detection.py: 人脸检测第一版
- face_detection2.py: 人脸检测第二版，加了中文支持
- face_detection3.py: 人脸检测第三版，加了人脸关键点检测，实现了点头摇头功能
- face_recog_dlib.py: 人脸检测第四版，通过dlib库加强识别准确度

- camera_test.py: K210图像识别硬件测试代码
- face_detect.py: K210人脸识别测试代码
- face_set_xy.py: K210人脸特征点检测测试代码
- test.py: K210通信测试代码

- shape_predictor_68_face_landmarks.dat: 人脸关键点检测预训练模型文件
- VC_redist.x64.exe: Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019
- dlib-19.24.1-cp311-cp311-win_amd64.whl: dlib库的预编译版本，用于Windows环境