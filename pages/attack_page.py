import cv2
import numpy as np
import dlib
import random
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QGridLayout
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
        
        # 随机噪声分组
        noise_group = QGroupBox("噪声对抗样本")
        noise_layout = QGridLayout()  # 改用网格布局
        noise_group.setLayout(noise_layout)
        
        # 定义不同的噪声对抗样本生成按钮
        self.generate_buttons = []
        noise_configs = [
            # 全局噪声按钮 - 格式: (按钮名称, 噪声强度std, 区域region)
            ("轻微噪声", 10, "global"),        # 轻微噪声
            ("中等噪声", 50, "global"),         # 中等噪声
            ("强烈噪声", 100, "global"),           # 强烈噪声
            ("极端噪声", 200, "global"),          # 极端噪声
            # 局部区域噪声按钮
            ("眼部噪声", 50, "eyes"),            # 眼部噪声
            ("嘴部噪声", 50, "mouth"),           # 嘴部噪声
            ("面部轮廓噪声", 50, "contour"),     # 面部轮廓噪声
            # 关键点噪声按钮
            ("关键点弱噪声", 50, "landmarks_weak"),    # 关键点弱噪声
            ("关键点强噪声", 150, "landmarks_strong"), # 关键点强噪声
            # 特殊区域噪声按钮
            ("眉毛噪声", 50, "eyebrows"),        # 眉毛噪声
            ("鼻子噪声", 50, "nose"),            # 鼻子噪声
            ("下巴噪声", 50, "chin")             # 下巴噪声
        ]
        
        for i, (name, std, region) in enumerate(noise_configs):
            btn = QPushButton(name)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 16px;
                    background-color: #FF5722;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 10px 20px;
                    margin: 5px;
                }
                QPushButton:hover {
                    background-color: #F4511E;
                }
            """)
            # 明确定义每个参数的作用，注意参数顺序：方法、区域、强度
            # 注意：generate_adversarial方法的签名是(self, mean, std, shape_str, ksize=3, weight=1.0)
            # 这里我们传递"Noise"作为mean参数，region作为shape_str参数，std作为std参数
            btn.clicked.connect(lambda checked, method="Noise", s=std, r=region: 
                              self.generate_adversarial(method, s, r))
            row, col = i // 3, i % 3  # 3列布局
            noise_layout.addWidget(btn, row, col)
            self.generate_buttons.append(btn)
        
        # FGSM分组
        fgsm_group = QGroupBox("FGSM对抗样本")
        fgsm_layout = QGridLayout()
        fgsm_group.setLayout(fgsm_layout)
        
        # 添加不同参数的FGSM按钮
        self.fgsm_buttons = []
        fgsm_configs = [
            ("FGSM弱", 0.01, None, 3, 0.5),     # 弱扰动
            ("FGSM中", 0.03, None, 3, 1.0),     # 中等扰动
            ("FGSM强", 0.05, None, 3, 1.5),     # 强扰动
            ("FGSM眼部", 0.03, "eyes", 5, 1.0), # 眼部区域
            ("FGSM面部", 0.03, "face", 5, 1.0), # 面部区域
            ("FGSM高对比", 0.1, "enhanced", 7, 2.5) # 高对比度增强版本
        ]
        
        for i, (name, epsilon, region, ksize, weight) in enumerate(fgsm_configs):
            btn = QPushButton(name)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 16px;
                    background-color: #9C27B0;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 10px 20px;
                    margin: 5px;
                }
                QPushButton:hover {
                    background-color: #7B1FA2;
                }
            """)
            # 使用元组保存FGSM的所有参数
            fgsm_params = ("FGSM", epsilon, region, ksize, weight)
            btn.clicked.connect(lambda checked, params=fgsm_params: 
                              self.generate_adversarial(*params))
            row, col = i // 2, i % 2
            fgsm_layout.addWidget(btn, row, col)
            self.fgsm_buttons.append(btn)
        
        # PGD分组
        pgd_group = QGroupBox("PGD对抗样本")
        pgd_layout = QGridLayout()
        pgd_group.setLayout(pgd_layout)
        
        # 添加不同参数的PGD按钮
        self.pgd_buttons = []
        pgd_configs = [
            ("PGD弱", 0.01, 5, 0.001),    # 弱扰动
            ("PGD中", 0.03, 10, 0.003),   # 中等扰动
            ("PGD强", 0.05, 20, 0.005),   # 强扰动
            ("PGD强+", 0.1, 30, 0.01)     # 超强扰动
        ]
        
        for i, (name, epsilon, iters, alpha) in enumerate(pgd_configs):
            btn = QPushButton(name)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 16px;
                    background-color: #3F51B5;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 10px 20px;
                    margin: 5px;
                }
                QPushButton:hover {
                    background-color: #303F9F;
                }
            """)
            # 使用元组保存PGD的所有参数
            pgd_params = ("PGD", epsilon, iters, alpha)
            btn.clicked.connect(lambda checked, params=pgd_params: 
                              self.generate_adversarial(*params))
            row, col = i // 2, i % 2
            pgd_layout.addWidget(btn, row, col)
            self.pgd_buttons.append(btn)
        
        # 图像变形分组
        transform_group = QGroupBox("图像变形对抗样本")
        transform_layout = QGridLayout()
        transform_group.setLayout(transform_layout)
        
        # 添加不同的图像变形按钮
        self.transform_buttons = []
        transform_configs = [
            ("轻微扭曲", "distort", 1),    # 轻微扭曲
            ("中等扭曲", "distort", 3),    # 中等扭曲
            ("旋转", "rotate", 3),         # 轻微旋转
            ("缩放", "scale", 0.95),       # 缩放
            ("眼部变形", "face_part", "eyes"),  # 眼部特定变形
            ("嘴部变形", "face_part", "mouth")  # 嘴部特定变形
        ]
        
        for i, (name, transform_type, param) in enumerate(transform_configs):
            btn = QPushButton(name)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 16px;
                    background-color: #009688;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 10px 20px;
                    margin: 5px;
                }
                QPushButton:hover {
                    background-color: #00796B;
                }
            """)
            # 使用元组保存变形的所有参数
            transform_params = ("Transform", transform_type, param)
            btn.clicked.connect(lambda checked, params=transform_params: 
                              self.generate_adversarial(*params))
            row, col = i // 2, i % 2
            transform_layout.addWidget(btn, row, col)
            self.transform_buttons.append(btn)
        
        middle_layout.addWidget(noise_group)
        middle_layout.addWidget(fgsm_group)
        middle_layout.addWidget(pgd_group)
        middle_layout.addWidget(transform_group)
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
                else:
                    raise Exception("无法识别到样本人脸")

                # 显示对抗样本识别结果
                self.display_landmarks(self.adversarial_landmarks, self.adversarial_result_label)
                # 计算并显示误差分析
                self.calculate_error()
            except Exception:
                self.adversarial_result_label.setText("无法识别到样本人脸")
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

    def generate_adversarial(self, mean, std, shape_str, ksize=3, weight=1.0):
        """
        生成对抗样本的主方法
        参数:
            mean: 生成方法类型，如"Noise"、"FGSM"、"PGD"、"Transform"
            std: 噪声强度或扰动大小
            shape_str: 区域参数，指定应用于哪个区域
            ksize: 卷积核大小（用于FGSM）
            weight: 权重参数（用于FGSM）
        """
        if self.original_image is not None and self.original_landmarks is not None:
            try:
                # 生成对抗样本
                self.adversarial_image = self.original_image.copy()
                
                print(f"生成对抗样本参数: method={mean}, std/epsilon={std}, region/shape={shape_str}")
                
                # 检查生成方法类型
                if mean == "FGSM":
                    self.adversarial_image = self.generate_fgsm_adversarial(std, shape_str, ksize, weight)
                elif mean == "PGD":
                    self.adversarial_image = self.generate_pgd_adversarial(std, shape_str, ksize)
                elif mean == "Transform":
                    self.adversarial_image = self.generate_transform_adversarial(std, shape_str)
                elif mean == "Noise":
                    # 直接将shape_str作为区域参数传递给噪声生成方法
                    self.adversarial_image = self.generate_noise_adversarial(std, shape_str)
                else:
                    # 原始的随机噪声方法（已废弃，保留代码作为参考）
                    # 根据shape_str确定噪声形状
                    if shape_str == "self.original_image.shape":
                        shape = self.original_image.shape
                    else:
                        shape = eval(shape_str)
                    
                    # 生成噪声
                    noise = np.random.normal(mean, std, shape).astype(np.uint8)
                    
                    # 如果噪声形状是2，则只在关键点周围添加噪声
                    if shape == 2:
                        # 获取所有关键点坐标
                        landmarks = []
                        for i in range(68):
                            x = self.original_landmarks.part(i).x
                            y = self.original_landmarks.part(i).y
                            landmarks.append((x, y))
                        
                        # 在关键点周围添加噪声
                        for x, y in landmarks:
                            if 0 <= x < self.adversarial_image.shape[1] and 0 <= y < self.adversarial_image.shape[0]:
                                self.adversarial_image[y, x] = cv2.add(self.adversarial_image[y, x], noise)
                    else:
                        # 全局添加噪声
                        self.adversarial_image = cv2.add(self.adversarial_image, noise)
                
                # 显示对抗样本
                frame_rgb = cv2.cvtColor(self.adversarial_image, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.adversarial_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                    self.adversarial_label.size(), Qt.KeepAspectRatio))

                # 存储对抗样本到MainWindow
                self.main_window.set_adversarial_image(self.adversarial_image)
            except Exception as e:
                # 捕获并显示错误信息
                error_message = f"生成对抗样本时发生错误: {str(e)}"
                print(error_message)
                self.adversarial_result_label.setText(error_message)
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

    def generate_fgsm_adversarial(self, epsilon, region=None, ksize=3, weight=1.0):
        """基于FGSM方法的对抗样本生成
        参数:
            epsilon: 扰动大小
            region: 扰动区域 (None: 全部, 'eyes': 眼部, 'face': 面部, 'enhanced': 增强模式)
            ksize: Sobel算子的内核大小，越大梯度越显著
            weight: 扰动权重，控制扰动强度
        返回:
            生成的对抗样本
        """
        # 图像预处理
        image = self.original_image.copy()
        image_float = image.astype(np.float32)
        
        # 特殊增强模式
        if region == "enhanced":
            # 增强图像对比度
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced_lab = cv2.merge((l, a, b))
            image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            image_float = image.astype(np.float32)
            
            # 同时应用多个方向的Sobel滤波增强边缘
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 使用更大的ksize获取更显著的梯度
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
            
            # 添加对角线方向的梯度以获得更丰富的扰动
            grad_xy = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=ksize)
            grad_yx = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=ksize)  # 修正：不能使用负数参数
            
            # 获取反方向的对角线梯度（用旋转90度的方式实现）
            rot_gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
            grad_alt = cv2.Sobel(rot_gray, cv2.CV_32F, 1, 1, ksize=ksize)
            grad_alt = cv2.rotate(grad_alt, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # 组合多个方向的梯度
            grad_magnitude = cv2.magnitude(grad_x, grad_y)
            grad_magnitude += 0.5 * (np.abs(grad_xy) + np.abs(grad_yx))
            
            # 非线性增强梯度
            grad_magnitude = np.power(grad_magnitude, 1.5)
            
            # 归一化梯度
            if np.max(grad_magnitude) > 0:
                grad_magnitude = grad_magnitude / np.max(grad_magnitude)
                
            # 基于图像内容自适应调整扰动强度
            # 在边缘和纹理区域增加扰动，在平滑区域减少扰动
            edge_mask = cv2.Canny(image, 50, 150)
            edge_mask = edge_mask.astype(np.float32) / 255.0
            edge_mask = cv2.dilate(edge_mask, np.ones((5, 5), np.uint8))
            
            # 添加额外的权重到人脸特征区域
            face_mask = np.zeros_like(gray, dtype=np.float32)
            
            # 给眼睛、鼻子、嘴巴区域添加更高权重
            feature_points = []
            # 眼睛区域
            for i in range(36, 48):
                feature_points.append((self.original_landmarks.part(i).x, self.original_landmarks.part(i).y))
            # 鼻子区域
            for i in range(27, 36):
                feature_points.append((self.original_landmarks.part(i).x, self.original_landmarks.part(i).y))
            # 嘴巴区域
            for i in range(48, 68):
                feature_points.append((self.original_landmarks.part(i).x, self.original_landmarks.part(i).y))
                
            for x, y in feature_points:
                cv2.circle(face_mask, (x, y), 10, 1.0, -1)
                
            # 合并所有掩码，最终扰动遵循自适应强度
            combined_mask = 0.7 + 0.3 * edge_mask + 0.5 * face_mask
            combined_mask = np.clip(combined_mask, 0, 2.0)
            
            # 生成扰动 (带符号，保留正负方向)
            perturbation = epsilon * np.sign(grad_magnitude) * weight * combined_mask
            
            # 扩展为3通道
            perturbation = np.stack([perturbation] * 3, axis=2)
            
            # 对原图像应用非线性变换增强对抗效果
            adversarial = np.tanh(image_float / 255.0 * 2 - 1) * 127.5 + 127.5
            
            # 添加扰动
            adversarial = adversarial + perturbation * 255.0
            
            # 确保像素值在有效范围内
            adversarial = np.clip(adversarial, 0, 255)
            
            return adversarial.astype(np.uint8)
        
        # 以下为原始FGSM实现
        # 简化的FGSM实现，使用图像梯度近似
        # 在实际深度学习模型中，这里应该用模型的输入梯度
        # 这里我们使用Sobel算子作为梯度的近似
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算x和y方向的梯度，使用更大的ksize增强梯度效果
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
        
        # 计算梯度幅度
        grad_magnitude = cv2.magnitude(grad_x, grad_y)
        
        # 归一化梯度
        if np.max(grad_magnitude) > 0:
            grad_magnitude = grad_magnitude / np.max(grad_magnitude)
        
        # 创建一个掩码图像
        mask = np.ones_like(grad_magnitude)
        
        # 如果指定了区域，只在该区域应用扰动
        if region == 'eyes':
            # 获取眼睛区域坐标
            left_eye_points = [(self.original_landmarks.part(i).x, self.original_landmarks.part(i).y) for i in range(36, 42)]
            right_eye_points = [(self.original_landmarks.part(i).x, self.original_landmarks.part(i).y) for i in range(42, 48)]
            
            # 创建眼睛区域的掩码
            mask = np.zeros_like(grad_magnitude)
            
            # 眼睛周围区域半径
            eye_radius = 30
            
            # 左眼中心
            left_eye_center = (
                sum(p[0] for p in left_eye_points) // len(left_eye_points),
                sum(p[1] for p in left_eye_points) // len(left_eye_points)
            )
            
            # 右眼中心
            right_eye_center = (
                sum(p[0] for p in right_eye_points) // len(right_eye_points),
                sum(p[1] for p in right_eye_points) // len(right_eye_points)
            )
            
            # 在掩码上绘制眼睛区域
            cv2.circle(mask, left_eye_center, eye_radius, 1, -1)
            cv2.circle(mask, right_eye_center, eye_radius, 1, -1)
            
        elif region == 'face':
            # 获取面部轮廓点
            face_points = [(self.original_landmarks.part(i).x, self.original_landmarks.part(i).y) for i in range(0, 17)]
            
            # 创建面部区域的掩码
            mask = np.zeros_like(grad_magnitude)
            
            # 将点转换为numpy数组
            face_points = np.array(face_points)
            
            # 创建一个凸包包含面部轮廓点
            hull = cv2.convexHull(face_points)
            
            # 在掩码上绘制面部区域
            cv2.fillConvexPoly(mask, hull, 1)
        
        # 将掩码扩展为三通道
        mask_3ch = np.stack([mask] * 3, axis=2)
        
        # 生成扰动 (FGSM使用梯度符号)
        perturbation = epsilon * np.sign(grad_magnitude) * weight
        
        # 扩展为3通道
        perturbation = np.stack([perturbation] * 3, axis=2)
        
        # 应用掩码，只在指定区域添加扰动
        perturbation = perturbation * mask_3ch
        
        # 添加扰动
        adversarial = image_float + perturbation * 255.0
        
        # 确保像素值在有效范围内
        adversarial = np.clip(adversarial, 0, 255)
        
        return adversarial.astype(np.uint8)

    def generate_pgd_adversarial(self, epsilon, iterations, alpha):
        """生成PGD (Projected Gradient Descent) 对抗样本
        参数:
            epsilon: 扰动大小上限
            iterations: 迭代次数
            alpha: 每次迭代的步长
        返回:
            生成的对抗样本
        """
        # 图像预处理
        image = self.original_image.copy()
        image_float = image.astype(np.float32) / 255.0
        
        # 初始化对抗样本，加一点随机初始化扰动
        delta = np.random.uniform(-epsilon, epsilon, image.shape) / 255.0
        x_adv = np.clip(image_float + delta, 0.0, 1.0)
        
        for _ in range(iterations):
            # 获取当前对抗样本的梯度
            # 这里简化实现，用Sobel算子代替实际模型梯度
            x_adv_uint8 = (x_adv * 255).astype(np.uint8)
            gray = cv2.cvtColor(x_adv_uint8, cv2.COLOR_BGR2GRAY)
            
            # 计算梯度
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            
            grad_magnitude = cv2.magnitude(grad_x, grad_y)
            
            # 归一化梯度
            if np.max(grad_magnitude) > 0:
                grad_magnitude = grad_magnitude / np.max(grad_magnitude)
            
            # 生成方向
            grad_sign = np.sign(grad_magnitude)
            
            # 扩展到3通道
            grad_sign = np.stack([grad_sign] * 3, axis=2)
            
            # 按梯度方向更新对抗样本
            x_adv = x_adv + alpha * grad_sign
            
            # 投影到原图像附近的epsilon球内
            delta = np.clip(x_adv - image_float, -epsilon, epsilon)
            x_adv = np.clip(image_float + delta, 0.0, 1.0)
        
        # 转回uint8格式
        adversarial = (x_adv * 255).astype(np.uint8)
        
        return adversarial
        
    def generate_transform_adversarial(self, transform_type, param):
        """生成基于几何变换的对抗样本
        参数:
            transform_type: 变换类型 (distort, rotate, scale, face_part)
            param: 变换参数
        返回:
            生成的对抗样本
        """
        image = self.original_image.copy()
        
        if transform_type == "distort":
            # 图像扭曲变形
            intensity = param  # 扭曲强度
            
            # 获取图像尺寸
            height, width = image.shape[:2]
            
            # 创建网格
            mapx, mapy = np.meshgrid(np.arange(width), np.arange(height))
            mapx = mapx.astype(np.float32)
            mapy = mapy.astype(np.float32)
            
            # 添加正弦扭曲
            mapx += intensity * np.sin(mapy/20)
            mapy += intensity * np.sin(mapx/20)
            
            # 应用扭曲
            distorted = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            
            return distorted
            
        elif transform_type == "rotate":
            # 图像旋转
            angle = param  # 旋转角度
            
            # 获取图像中心
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            # 计算旋转矩阵
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # 应用旋转
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
            
            return rotated
            
        elif transform_type == "scale":
            # 图像缩放
            scale = param  # 缩放因子
            
            # 获取图像尺寸
            height, width = image.shape[:2]
            
            # 计算新尺寸
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # 缩放图像
            scaled = cv2.resize(image, (new_width, new_height))
            
            # 创建新画布，确保输出尺寸与原始图像相同
            result = np.zeros_like(image)
            
            # 计算偏移量，使得图像居中
            offset_x = (width - new_width) // 2
            offset_y = (height - new_height) // 2
            
            # 将缩放后的图像放在原始画布中心
            result[offset_y:offset_y+new_height, offset_x:offset_x+new_width] = scaled
            
            return result
            
        elif transform_type == "face_part":
            # 特定人脸部位变形
            part = param  # 变形的部位
            
            # 创建一个变形网格
            height, width = image.shape[:2]
            mapx, mapy = np.meshgrid(np.arange(width), np.arange(height))
            mapx = mapx.astype(np.float32)
            mapy = mapy.astype(np.float32)
            
            if part == "eyes":
                # 眼部变形
                # 获取左右眼区域
                left_eye_points = [(self.original_landmarks.part(i).x, self.original_landmarks.part(i).y) for i in range(36, 42)]
                right_eye_points = [(self.original_landmarks.part(i).x, self.original_landmarks.part(i).y) for i in range(42, 48)]
                
                # 左眼中心
                left_eye_center = (
                    sum(p[0] for p in left_eye_points) // len(left_eye_points),
                    sum(p[1] for p in left_eye_points) // len(left_eye_points)
                )
                
                # 右眼中心
                right_eye_center = (
                    sum(p[0] for p in right_eye_points) // len(right_eye_points),
                    sum(p[1] for p in right_eye_points) // len(right_eye_points)
                )
                
                # 在眼睛周围添加局部扭曲
                for y in range(height):
                    for x in range(width):
                        # 计算点到左眼的距离
                        dl = np.sqrt((x - left_eye_center[0])**2 + (y - left_eye_center[1])**2)
                        # 计算点到右眼的距离
                        dr = np.sqrt((x - right_eye_center[0])**2 + (y - right_eye_center[1])**2)
                        
                        # 如果在眼睛附近，添加更强的扭曲
                        if dl < 30:
                            strength = (1 - dl/30) * 5
                            mapx[y, x] += strength * np.sin(y/10)
                            mapy[y, x] += strength * np.cos(x/10)
                        
                        if dr < 30:
                            strength = (1 - dr/30) * 5
                            mapx[y, x] += strength * np.sin(y/10)
                            mapy[y, x] += strength * np.cos(x/10)
                
            elif part == "mouth":
                # 嘴部变形
                # 获取嘴唇区域
                mouth_points = [(self.original_landmarks.part(i).x, self.original_landmarks.part(i).y) for i in range(48, 68)]
                
                # 嘴中心
                mouth_center = (
                    sum(p[0] for p in mouth_points) // len(mouth_points),
                    sum(p[1] for p in mouth_points) // len(mouth_points)
                )
                
                # 在嘴部周围添加局部扭曲
                for y in range(height):
                    for x in range(width):
                        # 计算点到嘴的距离
                        d = np.sqrt((x - mouth_center[0])**2 + (y - mouth_center[1])**2)
                        
                        # 如果在嘴部附近，添加扭曲
                        if d < 40:
                            strength = (1 - d/40) * 6
                            mapx[y, x] += strength * np.sin(x/8)
                            mapy[y, x] += strength * np.cos(y/8)
            
            # 应用变形
            result = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            
            return result
            
        return image  # 默认返回原图

    def generate_noise_adversarial(self, std, region):
        """生成针对特定区域的噪声对抗样本
        参数:
            std: 噪声标准差
            region: 噪声区域 (global, eyes, mouth, contour, landmarks_weak, landmarks_strong, eyebrows, nose, chin)
        返回:
            生成的对抗样本
        """
        try:
            image = self.original_image.copy()
            
            # 记录传入参数的类型和值
            print(f"噪声参数 - 强度(std): {std}, 类型: {type(std)}")
            print(f"噪声参数 - 区域(region): {region}, 类型: {type(region)}")
            
            # 确保std是数值类型
            if not isinstance(std, (int, float)):
                print(f"警告: 噪声强度参数类型错误, 值={std}, 类型={type(std)}")
                try:
                    std = float(std)
                except (ValueError, TypeError):
                    print(f"无法将噪声强度参数转换为数值，使用默认值3.0")
                    std = 3.0
                    
            # 确保region是字符串类型
            if not isinstance(region, str):
                print(f"警告: 区域参数类型错误, 值={region}, 类型={type(region)}")
                try:
                    region = str(region)
                except (ValueError, TypeError):
                    print(f"无法将区域参数转换为字符串，使用默认值'global'")
                    region = "global"
            
            # 有效区域类型列表
            valid_regions = ["global", "eyes", "mouth", "contour", "landmarks_weak", 
                            "landmarks_strong", "eyebrows", "nose", "chin"]
            
            # 验证区域参数是否有效
            if region not in valid_regions:
                print(f"警告: 未知的区域类型 '{region}'，有效区域类型为: {', '.join(valid_regions)}")
                print(f"将使用全局噪声作为替代")
                region = "global"
            
            # 创建噪声掩码
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.float32)
            
            # 根据区域类型设置掩码
            if region == "global":
                # 全局噪声
                mask = np.ones((height, width), dtype=np.float32)
                
            elif region == "eyes":
                # 眼部区域噪声
                # 左眼关键点 (36-41)
                left_eye_points = [(self.original_landmarks.part(i).x, self.original_landmarks.part(i).y) for i in range(36, 42)]
                # 右眼关键点 (42-47)
                right_eye_points = [(self.original_landmarks.part(i).x, self.original_landmarks.part(i).y) for i in range(42, 48)]
                
                # 左眼中心
                left_eye_center = (
                    sum(p[0] for p in left_eye_points) // len(left_eye_points),
                    sum(p[1] for p in left_eye_points) // len(left_eye_points)
                )
                
                # 右眼中心
                right_eye_center = (
                    sum(p[0] for p in right_eye_points) // len(right_eye_points),
                    sum(p[1] for p in right_eye_points) // len(right_eye_points)
                )
                
                # 眼睛周围区域半径
                eye_radius = 30
                
                # 在掩码上绘制眼睛区域
                cv2.circle(mask, left_eye_center, eye_radius, 1, -1)
                cv2.circle(mask, right_eye_center, eye_radius, 1, -1)
                
            elif region == "mouth":
                # 嘴部区域噪声
                # 嘴部关键点 (48-67)
                mouth_points = [(self.original_landmarks.part(i).x, self.original_landmarks.part(i).y) for i in range(48, 68)]
                
                # 将点转换为numpy数组
                mouth_points = np.array(mouth_points)
                
                # 创建一个凸包包含嘴部轮廓点
                hull = cv2.convexHull(mouth_points)
                
                # 在掩码上绘制嘴部区域，并稍微扩大区域
                cv2.fillConvexPoly(mask, hull, 1)
                kernel = np.ones((15, 15), np.uint8)
                mask = cv2.dilate(mask, kernel)
                
            elif region == "contour":
                # 面部轮廓噪声
                # 面部轮廓关键点 (0-16)
                contour_points = [(self.original_landmarks.part(i).x, self.original_landmarks.part(i).y) for i in range(0, 17)]
                
                # 将点转换为numpy数组
                contour_points = np.array(contour_points)
                
                # 创建一个凸包包含面部轮廓点
                hull = cv2.convexHull(contour_points)
                
                # 在掩码上绘制面部轮廓区域
                cv2.fillConvexPoly(mask, hull, 1)
                
            elif region == "landmarks_weak" or region == "landmarks_strong":
                # 关键点周围噪声
                # 获取所有关键点坐标
                landmark_points = []
                for i in range(68):
                    x = self.original_landmarks.part(i).x
                    y = self.original_landmarks.part(i).y
                    landmark_points.append((x, y))
                
                # 关键点影响半径
                radius = 5 if region == "landmarks_weak" else 10
                
                # 在每个关键点周围添加噪声区域
                for x, y in landmark_points:
                    cv2.circle(mask, (x, y), radius, 1, -1)
                    
            elif region == "eyebrows":
                # 眉毛区域噪声
                # 左眉毛关键点 (17-21)
                left_eyebrow_points = [(self.original_landmarks.part(i).x, self.original_landmarks.part(i).y) for i in range(17, 22)]
                # 右眉毛关键点 (22-26)
                right_eyebrow_points = [(self.original_landmarks.part(i).x, self.original_landmarks.part(i).y) for i in range(22, 27)]
                
                # 将点转换为numpy数组
                left_eyebrow_points = np.array(left_eyebrow_points)
                right_eyebrow_points = np.array(right_eyebrow_points)
                
                # 在掩码上绘制眉毛区域
                cv2.fillConvexPoly(mask, cv2.convexHull(left_eyebrow_points), 1)
                cv2.fillConvexPoly(mask, cv2.convexHull(right_eyebrow_points), 1)
                # 扩大区域
                kernel = np.ones((10, 10), np.uint8)
                mask = cv2.dilate(mask, kernel)
                
            elif region == "nose":
                # 鼻子区域噪声
                # 鼻子关键点 (27-35)
                nose_points = [(self.original_landmarks.part(i).x, self.original_landmarks.part(i).y) for i in range(27, 36)]
                
                # 将点转换为numpy数组
                nose_points = np.array(nose_points)
                
                # 在掩码上绘制鼻子区域
                cv2.fillConvexPoly(mask, cv2.convexHull(nose_points), 1)
                # 扩大区域
                kernel = np.ones((10, 10), np.uint8)
                mask = cv2.dilate(mask, kernel)
                
            elif region == "chin":
                # 下巴区域噪声
                # 下巴关键点 (6-10)
                chin_points = [(self.original_landmarks.part(i).x, self.original_landmarks.part(i).y) for i in range(6, 11)]
                
                # 增加一些辅助点使区域更合理
                chin_center = self.original_landmarks.part(8)
                chin_bottom = (chin_center.x, chin_center.y + 20)
                chin_left = (chin_points[0][0] - 10, chin_points[0][1] + 10)
                chin_right = (chin_points[-1][0] + 10, chin_points[-1][1] + 10)
                
                chin_points.append(chin_bottom)
                chin_points.append(chin_left)
                chin_points.append(chin_right)
                
                # 将点转换为numpy数组
                chin_points = np.array(chin_points)
                
                # 在掩码上绘制下巴区域
                cv2.fillConvexPoly(mask, cv2.convexHull(chin_points), 1)
            
            # 生成噪声
            noise = np.random.normal(0, std, image.shape).astype(np.float32)
            
            # 将掩码扩展为三通道
            mask_3ch = np.stack([mask] * 3, axis=2)
            
            # 应用掩码，只在指定区域添加噪声
            noise = noise * mask_3ch
            
            # 将图像转为float32以防止溢出
            image_float = image.astype(np.float32)
            
            # 添加噪声
            adversarial = image_float + noise
            
            # 确保像素值在有效范围内
            adversarial = np.clip(adversarial, 0, 255)
            
            return adversarial.astype(np.uint8)
            
        except Exception as e:
            # 捕获异常，打印错误信息并返回原始图像
            print(f"生成噪声对抗样本时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return image