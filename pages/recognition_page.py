import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class RecognitionPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()
        self.setup_camera()

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

    def setup_camera(self):
        self.camera = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms 刷新率

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.camera_label.size(), Qt.KeepAspectRatio))

    def capture_frame(self):
        ret, frame = self.camera.read()
        if ret:
            self.captured_frame = frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

            # 存储捕获的图像到MainWindow
            self.main_window.set_captured_image(frame)

    def start_recognition(self):
        if self.captured_frame is not None:
            # TODO: 实现物体识别逻辑
            x, y = np.random.randint(0, 100, 2)
            self.result_label.setText(f"识别结果:\n物体坐标: ({x}, {y})")
        else:
            self.result_label.setText("请先捕获图像")

    def closeEvent(self, event):
        self.camera.release()
        super().closeEvent(event)