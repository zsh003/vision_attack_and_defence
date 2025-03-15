import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap


class AttackPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)

        # 左侧原始图像显示区域
        left_layout = QVBoxLayout()
        self.original_label = QLabel("原始图像")
        self.original_label.setMinimumSize(480, 360)
        self.original_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.original_label)

        # 中间控制区域
        middle_layout = QVBoxLayout()
        self.generate_btn = QPushButton("生成对抗样本")
        self.generate_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                background-color: #FF5722;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #F4511E;
            }
        """)
        self.generate_btn.clicked.connect(self.generate_adversarial)
        middle_layout.addWidget(self.generate_btn)
        middle_layout.addStretch()

        # 右侧对抗样本显示区域
        right_layout = QVBoxLayout()
        self.adversarial_label = QLabel("对抗样本")
        self.adversarial_label.setMinimumSize(480, 360)
        self.adversarial_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.adversarial_label)

        # 识别按钮和结果显示
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

        self.result_label = QLabel()
        right_layout.addWidget(self.result_label)

        layout.addLayout(left_layout)
        layout.addLayout(middle_layout)
        layout.addLayout(right_layout)

        self.original_image = None
        self.adversarial_image = None

    def showEvent(self, event):
        super().showEvent(event)
        self.load_original_image()

    def load_original_image(self):
        self.original_image = self.main_window.get_captured_image()
        if self.original_image is not None:
            frame_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.original_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.original_label.size(), Qt.KeepAspectRatio))

    def generate_adversarial(self):
        if self.original_image is not None:
            # TODO: 实现对抗样本生成逻辑
            # 简单的图像扰动
            self.adversarial_image = self.original_image.copy()
            noise = np.random.normal(0, 25, self.original_image.shape).astype(np.uint8)
            self.adversarial_image = cv2.add(self.adversarial_image, noise)

            frame_rgb = cv2.cvtColor(self.adversarial_image, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.adversarial_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.adversarial_label.size(), Qt.KeepAspectRatio))

            # 存储对抗样本到MainWindow
            self.main_window.set_adversarial_image(self.adversarial_image)
        else:
            self.result_label.setText("请先在识别页面捕获图像")

    def start_recognition(self):
        if self.adversarial_image is not None:
            # TODO: 实现物体识别逻辑
            x, y = np.random.randint(0, 100, 2)
            self.result_label.setText(f"识别结果:\n物体坐标: ({x}, {y})")
        else:
            self.result_label.setText("请先生成对抗样本")