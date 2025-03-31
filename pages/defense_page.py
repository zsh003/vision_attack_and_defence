import cv2
import numpy as np
import dlib
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
        self.setup_face_detector()

    def setup_face_detector(self):
        # 加载dlib的人脸检测器和预测器
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(predictor_path)

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 原始对抗样本显示
        self.original_label = QLabel("原始对抗样本")
        self.original_label.setMinimumSize(480, 360)
        self.original_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.original_label)

        # 原始对抗样本识别结果显示
        self.original_result_label = QLabel("原始对抗样本识别结果")
        self.original_result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.original_result_label)

        # 创建滚动区域用于显示多个防御结果
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.defense_layout = QHBoxLayout(scroll_content)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        self.adversarial_image = None
        self.original_landmarks = None
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

            # 识别结果显示
            result_label = QLabel()
            result_label.setAlignment(Qt.AlignCenter)
            v_layout.addWidget(result_label)

            # 误差分析显示
            error_label = QLabel()
            error_label.setAlignment(Qt.AlignCenter)
            v_layout.addWidget(error_label)

            self.defense_widgets.append({
                'widget': widget,
                'image_label': image_label,
                'result_label': result_label,
                'error_label': error_label,
                'method': method
            })
            self.defense_layout.addWidget(widget)


    def start_recognition(self, method):
        # 对原始对抗样本进行人脸检测和关键点提取
        gray = cv2.cvtColor(self.adversarial_image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) > 0:
            self.original_landmarks = self.predictor(gray, faces[0])
            # 显示原始对抗样本识别结果
            self.display_landmarks(self.original_landmarks, self.original_result_label)

        # 应用所有防御方法
        for widget_info in self.defense_widgets:
            processed_image = widget_info['method'].process_func(self.adversarial_image)
            
            # 显示处理后的图像
            frame_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            widget_info['image_label'].setPixmap(QPixmap.fromImage(qt_image).scaled(
                widget_info['image_label'].size(), Qt.KeepAspectRatio))
            
            # 对处理后的图像进行人脸检测和关键点提取
            gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            if len(faces) > 0:
                landmarks = self.predictor(gray, faces[0])
                # 显示识别结果
                self.display_landmarks(landmarks, widget_info['result_label'])
                # 计算并显示误差分析
                self.calculate_error(self.original_landmarks, landmarks, widget_info['error_label'])
            else:
                widget_info['result_label'].setText("未检测到人脸")
                widget_info['error_label'].setText("无法计算误差")
                    
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

            

    def display_landmarks(self, landmarks, label):
        if landmarks is not None:
            result_text = "人脸关键点坐标：\n"
            
            # 眼睛坐标
            left_eye = ((landmarks.part(36).x + landmarks.part(39).x) // 2,
                       (landmarks.part(36).y + landmarks.part(39).y) // 2)
            right_eye = ((landmarks.part(42).x + landmarks.part(45).x) // 2,
                        (landmarks.part(42).y + landmarks.part(45).y) // 2)
            result_text += f"左眼中心: ({left_eye[0]}, {left_eye[1]})\n"
            result_text += f"右眼中心: ({right_eye[0]}, {right_eye[1]})\n"
            
            # 鼻子坐标
            nose = (landmarks.part(30).x, landmarks.part(30).y)
            result_text += f"鼻尖: ({nose[0]}, {nose[1]})\n"
            
            # 嘴巴坐标
            mouth_left = (landmarks.part(48).x, landmarks.part(48).y)
            mouth_right = (landmarks.part(54).x, landmarks.part(54).y)
            mouth_center = ((mouth_left[0] + mouth_right[0]) // 2,
                          (mouth_left[1] + mouth_right[1]) // 2)
            result_text += f"嘴巴中心: ({mouth_center[0]}, {mouth_center[1]})\n"
            
            # 下巴坐标
            chin = (landmarks.part(8).x, landmarks.part(8).y)
            result_text += f"下巴: ({chin[0]}, {chin[1]})"
            
            label.setText(result_text)

    def calculate_error(self, orig_landmarks, processed_landmarks, label):
        if orig_landmarks is not None and processed_landmarks is not None:
            error_text = "误差分析：\n"
            
            # 计算各个关键点的误差
            features = {
                '左眼中心': [(36, 39), (36, 39)],
                '右眼中心': [(42, 45), (42, 45)],
                '鼻尖': [(30, 30), (30, 30)],
                '嘴巴中心': [(48, 54), (48, 54)],
                '下巴': [(8, 8), (8, 8)]
            }
            
            total_error = 0
            for feature, points in features.items():
                orig_x = (orig_landmarks.part(points[0][0]).x + 
                         orig_landmarks.part(points[0][1]).x) // 2
                orig_y = (orig_landmarks.part(points[0][0]).y + 
                         orig_landmarks.part(points[0][1]).y) // 2
                proc_x = (processed_landmarks.part(points[1][0]).x + 
                         processed_landmarks.part(points[1][1]).x) // 2
                proc_y = (processed_landmarks.part(points[1][0]).y + 
                         processed_landmarks.part(points[1][1]).y) // 2
                
                error = np.sqrt((orig_x - proc_x)**2 + (orig_y - proc_y)**2)
                total_error += error
                error_text += f"{feature}误差: {error:.2f}像素\n"
            
            avg_error = total_error / len(features)
            error_text += f"\n平均误差: {avg_error:.2f}像素"
            
            label.setText(error_text)

    def gaussian_defense(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def median_defense(self, image):
        return cv2.medianBlur(image, 5)

    def bilateral_defense(self, image):
        return cv2.bilateralFilter(image, 9, 75, 75)

    
