import cv2
import numpy as np
import dlib
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class RecognitionPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()
        self.setup_camera()
        self.setup_face_detector()
        
    def setup_face_detector(self):
        # 加载dlib的人脸检测器和预测器
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(predictor_path)

    def init_ui(self):
        layout = QHBoxLayout(self)

        # 左侧摄像头显示区域
        left_layout = QVBoxLayout()
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.camera_label)

        # 右侧控制和显示区域
        right_layout = QVBoxLayout()

        # 捕获按钮
        self.capture_btn = QPushButton("捕获")
        self.capture_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.capture_btn.clicked.connect(self.capture_frame)
        right_layout.addWidget(self.capture_btn)

        # 捕获图像显示
        self.captured_label = QLabel("捕获的图像")
        self.captured_label.setMinimumSize(320, 240)
        self.captured_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.captured_label)

        # 图像参数显示
        self.info_label = QLabel()
        right_layout.addWidget(self.info_label)

        # 识别按钮
        self.recognize_btn = QPushButton("开始识别")
        self.recognize_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.recognize_btn.clicked.connect(self.start_recognition)
        right_layout.addWidget(self.recognize_btn)

        # 识别结果显示
        self.result_label = QLabel()
        right_layout.addWidget(self.result_label)

        right_layout.addStretch()

        layout.addLayout(left_layout, stretch=2)
        layout.addLayout(right_layout, stretch=1)

        self.captured_frame = None
        self.landmarks = None

    def setup_camera(self):
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms 刷新率

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            # 实时人脸检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            
            # 绘制人脸框和关键点
            for face in faces:
                x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                landmarks = self.predictor(gray, face)
                for n in range(68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.camera_label.size(), Qt.KeepAspectRatio))

    def capture_frame(self):
        ret, frame = self.camera.read()
        if ret:
            self.captured_frame = frame.copy()  # 保存原始帧的副本，不含标记
            
            # 进行人脸检测和关键点提取
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            if len(faces) > 0:
                self.landmarks = self.predictor(gray, faces[0])
                # 不再在图像上绘制关键点
            
            # 显示原始图像，不包含标记
            frame_rgb = cv2.cvtColor(self.captured_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.captured_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.captured_label.size(), Qt.KeepAspectRatio))

            # 显示图像参数
            info_text = f"图像尺寸: {w}x{h}\n"
            info_text += f"通道数: {ch}\n"
            info_text += f"数据类型: {frame.dtype}"
            self.info_label.setText(info_text)

            # 存储捕获的图像到MainWindow（使用无标记的原始图像）
            self.main_window.set_captured_image(self.captured_frame)

    def start_recognition(self):
        if self.captured_frame is not None and self.landmarks is not None:
            # 提取关键点坐标
            result_text = "人脸关键点坐标：\n"
            
            # 眼睛坐标
            left_eye = ((self.landmarks.part(36).x + self.landmarks.part(39).x) // 2,
                       (self.landmarks.part(36).y + self.landmarks.part(39).y) // 2)
            right_eye = ((self.landmarks.part(42).x + self.landmarks.part(45).x) // 2,
                        (self.landmarks.part(42).y + self.landmarks.part(45).y) // 2)
            result_text += f"左眼中心: ({left_eye[0]}, {left_eye[1]})\n"
            result_text += f"右眼中心: ({right_eye[0]}, {right_eye[1]})\n"
            
            # 鼻子坐标
            nose = (self.landmarks.part(30).x, self.landmarks.part(30).y)
            result_text += f"鼻尖: ({nose[0]}, {nose[1]})\n"
            
            # 嘴巴坐标
            mouth_left = (self.landmarks.part(48).x, self.landmarks.part(48).y)
            mouth_right = (self.landmarks.part(54).x, self.landmarks.part(54).y)
            mouth_center = ((mouth_left[0] + mouth_right[0]) // 2,
                          (mouth_left[1] + mouth_right[1]) // 2)
            result_text += f"嘴巴中心: ({mouth_center[0]}, {mouth_center[1]})\n"
            
            # 下巴坐标
            chin = (self.landmarks.part(8).x, self.landmarks.part(8).y)
            result_text += f"下巴: ({chin[0]}, {chin[1]})"
            
            self.result_label.setText(result_text)
        else:
            self.result_label.setText("请先捕获包含人脸的图像")

    def closeEvent(self, event):
        self.camera.release()
        super().closeEvent(event)