import cv2
import numpy as np
import dlib
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap


class AttackPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()
        self.setup_face_detector()

    def setup_face_detector(self):
        # 加载dlib的人脸检测器和预测器
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(predictor_path)

    def init_ui(self):
        layout = QHBoxLayout(self)

        # 左侧原始图像显示区域
        left_layout = QVBoxLayout()
        self.original_label = QLabel("原始图像")
        self.original_label.setMinimumSize(480, 360)
        self.original_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.original_label)

        # 原始图像识别结果显示
        self.original_result_label = QLabel("原始图像识别结果")
        self.original_result_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.original_result_label)

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
        self.adversarial_result_label = QLabel("对抗样本识别结果")
        self.error_analysis_label = QLabel("误差分析")
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
        right_layout.addWidget(self.adversarial_result_label)
        right_layout.addWidget(self.error_analysis_label)

        layout.addLayout(left_layout)
        layout.addLayout(middle_layout)
        layout.addLayout(right_layout)

        self.original_image = None
        self.adversarial_image = None
        self.original_landmarks = None
        self.adversarial_landmarks = None

    def showEvent(self, event):
        super().showEvent(event)
        self.load_original_image()
    
    def start_recognition(self):
        if self.adversarial_image is not None:
            
            try:
                # 对对抗样本进行人脸检测和关键点提取
                gray = cv2.cvtColor(self.adversarial_image, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)
                if len(faces) > 0:
                    self.adversarial_landmarks = self.predictor(gray, faces[0])

                # 显示对抗样本识别结果
                self.display_landmarks(self.adversarial_landmarks, self.adversarial_result_label)
                # 计算并显示误差分析
                self.calculate_error()
            except:
                self.adversarial_result_label.setText("对抗样本无法识别")
                self.error_analysis_label.setText("")
            
        else:
            self.adversarial_result_label.setText("请先生成对抗样本!")
            self.error_analysis_label.setText("")

    def load_original_image(self):
        self.original_image = self.main_window.get_captured_image()
        if self.original_image is not None:
            frame_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.original_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.original_label.size(), Qt.KeepAspectRatio))

            # 进行人脸检测和关键点提取
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            if len(faces) > 0:
                self.original_landmarks = self.predictor(gray, faces[0])
                # 显示原始图像识别结果
                self.display_landmarks(self.original_landmarks, self.original_result_label)
        else:
            self.original_result_label.setText("请先在识别页面捕获图像")

    def generate_adversarial(self):
        if self.original_image is not None and self.original_landmarks is not None:
            # 生成对抗样本

            # 直接在原图上添加噪声
            self.adversarial_image = self.original_image.copy()
            #noise = np.random.normal(0, 25, self.original_image.shape).astype(np.uint8) # 识别不到人脸
            #noise = np.random.normal(0, 5, 2).astype(np.uint8) # 误差很小 1像素
            #noise = np.random.normal(0, 25, 2).astype(np.uint8) # 误差很小 1像素
            #noise = np.random.normal(0, 100, 2).astype(np.uint8) # 误差稍小 3-4像素 效果可以
            #noise = np.random.normal(20, 50, 2).astype(np.uint8) # 这个效果也可以
            noise = np.random.normal(0, 0.75, self.original_image.shape).astype(np.uint8) # 演示最佳，scale=1会有点模糊了

            print(self.original_image.shape)

            self.adversarial_image = cv2.add(self.adversarial_image, noise)

            #在关键点周围添加扰动
            # for n in range(68):
            #     x = self.original_landmarks.part(n).x
            #     y = self.original_landmarks.part(n).y
            #     # 在关键点周围添加随机扰动
            #     noise = np.random.normal(0, 5, 2).astype(np.int32)
            #     cv2.circle(self.adversarial_image, (x + noise[0], y + noise[1]), 2, (0, 0, 255), -1)
            
            # 显示对抗样本
            frame_rgb = cv2.cvtColor(self.adversarial_image, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.adversarial_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.adversarial_label.size(), Qt.KeepAspectRatio))

            # 存储对抗样本到MainWindow
            self.main_window.set_adversarial_image(self.adversarial_image)
        else:
            self.adversarial_result_label.setText("请先在识别页面捕获包含人脸的图像")

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
            
    def calculate_error(self):
        if self.original_landmarks is not None and self.adversarial_landmarks is not None:
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
                orig_x = (self.original_landmarks.part(points[0][0]).x + 
                         self.original_landmarks.part(points[0][1]).x) // 2
                orig_y = (self.original_landmarks.part(points[0][0]).y + 
                         self.original_landmarks.part(points[0][1]).y) // 2
                adv_x = (self.adversarial_landmarks.part(points[1][0]).x + 
                        self.adversarial_landmarks.part(points[1][1]).x) // 2
                adv_y = (self.adversarial_landmarks.part(points[1][0]).y + 
                        self.adversarial_landmarks.part(points[1][1]).y) // 2
                
                error = np.sqrt((orig_x - adv_x)**2 + (orig_y - adv_y)**2)
                total_error += error
                error_text += f"{feature}误差: {error:.2f}像素\n"
            
            avg_error = total_error / len(features)
            error_text += f"\n平均误差: {avg_error:.2f}像素"
            
            self.error_analysis_label.setText(error_text)