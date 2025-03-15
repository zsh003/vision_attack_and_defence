import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap


class DefenseMethod:
    def __init__(self, name, process_func):
        self.name = name
        self.process_func = process_func


class DefensePage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()
        self.setup_defense_methods()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 原始对抗样本显示
        self.original_label = QLabel("原始对抗样本")
        self.original_label.setMinimumSize(480, 360)
        self.original_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.original_label)

        # 创建滚动区域用于显示多个防御结果
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.defense_layout = QHBoxLayout(scroll_content)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        self.adversarial_image = None
        self.defense_widgets = []

    def setup_defense_methods(self):
        # 定义防御方法
        self.defense_methods = [
            DefenseMethod("高斯滤波", self.gaussian_defense),
            DefenseMethod("中值滤波", self.median_defense),
            DefenseMethod("双边滤波", self.bilateral_defense)
        ]

        # 为每个防御方法创建显示组件
        for method in self.defense_methods:
            widget = QWidget()
            v_layout = QVBoxLayout(widget)

            # 标题
            title = QLabel(method.name)
            title.setAlignment(Qt.AlignCenter)
            v_layout.addWidget(title)

            # 图像显示
            image_label = QLabel()
            image_label.setMinimumSize(320, 240)
            image_label.setAlignment(Qt.AlignCenter)
            v_layout.addWidget(image_label)

            # 识别按钮
            recognize_btn = QPushButton("开始识别")
            recognize_btn.setStyleSheet("""
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
            recognize_btn.clicked.connect(lambda checked, m=method: self.start_recognition(m))
            v_layout.addWidget(recognize_btn)

            # 结果显示
            result_label = QLabel()
            v_layout.addWidget(result_label)

            self.defense_widgets.append({
                'widget': widget,
                'image_label': image_label,
                'result_label': result_label,
                'method': method
            })
            self.defense_layout.addWidget(widget)

    def showEvent(self, event):
        super().showEvent(event)
        self.load_adversarial_image()

    def load_adversarial_image(self):
        self.adversarial_image = self.main_window.get_adversarial_image()
        if self.adversarial_image is not None:
            # 显示原始对抗样本
            frame_rgb = cv2.cvtColor(self.adversarial_image, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.original_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.original_label.size(), Qt.KeepAspectRatio))

            # 应用所有防御方法
            for widget_info in self.defense_widgets:
                processed_image = widget_info['method'].process_func(self.adversarial_image)
                frame_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                widget_info['image_label'].setPixmap(QPixmap.fromImage(qt_image).scaled(
                    widget_info['image_label'].size(), Qt.KeepAspectRatio))
                widget_info['result_label'].clear()

    def gaussian_defense(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def median_defense(self, image):
        return cv2.medianBlur(image, 5)

    def bilateral_defense(self, image):
        return cv2.bilateralFilter(image, 9, 75, 75)

    def start_recognition(self, method):
        for widget_info in self.defense_widgets:
            if widget_info['method'] == method:
                # TODO: 实现物体识别逻辑
                x, y = np.random.randint(0, 100, 2)
                widget_info['result_label'].setText(f"识别结果:\n物体坐标: ({x}, {y})")
                break