import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea, QTabWidget, QGridLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap


class DefenseMethod:
    def __init__(self, name, process_func):
        self.name = name
        self.process_func = process_func


class ComparisonChart(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(ComparisonChart, self).__init__(self.fig)
        self.fig.tight_layout()
        
    def plot_comparison(self, original_data, adversarial_data, defense_data_list, feature_names, method_names):
        """
        绘制误差对比图表
        
        参数:
            original_data: 原始图像关键点坐标
            adversarial_data: 对抗样本关键点坐标
            defense_data_list: 防御方法处理后的关键点坐标列表
            feature_names: 特征名称列表
            method_names: 防御方法名称列表
        """
        self.axes.clear()
        
        # 计算每个方法相对于原始图像的误差
        original_vs_adversarial = self._calculate_differences(original_data, adversarial_data)
        
        defense_errors = []
        for defense_data in defense_data_list:
            if defense_data is not None:
                errors = self._calculate_differences(original_data, defense_data)
                defense_errors.append(errors)
            else:
                defense_errors.append([np.nan] * len(feature_names))
        
        # 设置x轴位置
        x = np.arange(len(feature_names))
        width = 0.15  # 柱状图宽度
        
        # 绘制对抗样本误差
        self.axes.bar(x, original_vs_adversarial, width, label='对抗样本')
        
        # 绘制各防御方法误差
        for i, errors in enumerate(defense_errors):
            self.axes.bar(x + width * (i + 1), errors, width, label=method_names[i])
        
        # 设置图表属性
        self.axes.set_xlabel('面部特征')
        self.axes.set_ylabel('相对于原始图像的误差 (像素)')
        self.axes.set_title('防御方法误差对比')
        self.axes.set_xticks(x + width * (len(defense_errors) / 2))
        self.axes.set_xticklabels(feature_names)
        self.axes.legend()
        
        # 添加中文标题
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
        self.fig.suptitle('各防御方法与原始图像的误差对比', fontsize=14)
        
        # 添加图表说明
        if len(defense_errors) > 0:
            explanation_text = "说明: 柱状图高度表示误差大小，越低表示防御效果越好"
            self.fig.text(0.5, 0.01, explanation_text, ha='center', fontsize=10)
        
        self.fig.tight_layout()
        self.draw()
    
    def _calculate_differences(self, coords1, coords2):
        """计算两组坐标之间的欧氏距离"""
        differences = []
        for (x1, y1), (x2, y2) in zip(coords1, coords2):
            diff = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            differences.append(diff)
        return differences


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
        
        # 创建标签页
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # 图像对比标签页
        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)
        tabs.addTab(image_tab, "图像对比")
        
        # 图表对比标签页
        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)
        tabs.addTab(chart_tab, "误差图表")
        
        # 在图像对比标签页中添加原始图像和对抗样本显示区域
        images_grid = QGridLayout()
        image_layout.addLayout(images_grid)
        
        # 原始图像显示
        self.original_image_label = QLabel("原始图像")
        self.original_image_label.setMinimumSize(320, 240)
        self.original_image_label.setAlignment(Qt.AlignCenter)
        images_grid.addWidget(self.original_image_label, 0, 0)
        
        # 原始图像识别结果显示
        self.original_result_label = QLabel("原始图像识别结果")
        self.original_result_label.setMinimumSize(320, 100)
        self.original_result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        images_grid.addWidget(self.original_result_label, 1, 0)
        
        # 对抗样本显示
        self.adversarial_label = QLabel("对抗样本")
        self.adversarial_label.setMinimumSize(320, 240)
        self.adversarial_label.setAlignment(Qt.AlignCenter)
        images_grid.addWidget(self.adversarial_label, 0, 1)
        
        # 对抗样本识别结果显示
        self.adversarial_result_label = QLabel("对抗样本识别结果")
        self.adversarial_result_label.setMinimumSize(320, 100)
        self.adversarial_result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        images_grid.addWidget(self.adversarial_result_label, 1, 1)
        
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
        recognize_btn.clicked.connect(self.start_recognition)
        image_layout.addWidget(recognize_btn)
        
        # 创建滚动区域用于显示多个防御结果
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.defense_layout = QHBoxLayout(scroll_content)
        scroll.setWidget(scroll_content)
        image_layout.addWidget(scroll)
        
        # 在图表标签页中添加图表
        self.error_chart = ComparisonChart(width=8, height=5, dpi=100)
        chart_layout.addWidget(self.error_chart)
        
        self.error_details_label = QLabel("请先进行识别以生成误差数据")
        self.error_details_label.setAlignment(Qt.AlignCenter)
        chart_layout.addWidget(self.error_details_label)
        
        self.original_image = None
        self.adversarial_image = None
        self.original_landmarks = None
        self.adversarial_landmarks = None
        self.defense_widgets = []
        self.feature_names = ['左眼中心', '右眼中心', '鼻尖', '嘴巴中心', '下巴']
        self.all_landmarks_data = {
            'original': None,
            'adversarial': None,
            'defenses': {}
        }

    def setup_defense_methods(self):
        # 定义防御方法
        self.defense_methods = [
            DefenseMethod("高斯滤波", self.gaussian_defense),
            DefenseMethod("中值滤波", self.median_defense),
            DefenseMethod("双边滤波", self.bilateral_defense),
            DefenseMethod("JPEG压缩", self.jpeg_defense),
            DefenseMethod("直方图均衡化", self.histogram_eq_defense),
            DefenseMethod("随机抖动", self.random_noise_defense),
            DefenseMethod("边缘增强", self.edge_enhance_defense),
            DefenseMethod("色彩空间变换", self.color_space_defense)
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

            # 识别结果显示
            result_label = QLabel()
            result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            v_layout.addWidget(result_label)

            # 误差分析显示
            error_label = QLabel()
            error_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            v_layout.addWidget(error_label)

            self.defense_widgets.append({
                'widget': widget,
                'image_label': image_label,
                'result_label': result_label,
                'error_label': error_label,
                'method': method
            })
            self.defense_layout.addWidget(widget)


    def start_recognition(self):
        if self.adversarial_image is None or self.original_image is None:
            return
            
        # 对原始图像进行人脸检测和关键点提取
        gray_original = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        faces_original = self.detector(gray_original)
        if len(faces_original) > 0:
            self.original_landmarks = self.predictor(gray_original, faces_original[0])
            # 显示原始图像识别结果
            self.display_landmarks(self.original_landmarks, self.original_result_label)
            
            # 存储原始关键点数据
            self.all_landmarks_data['original'] = self.extract_landmark_coords(self.original_landmarks)
        else:
            self.original_result_label.setText("未检测到人脸")
            self.all_landmarks_data['original'] = None
            
        # 对对抗样本进行人脸检测和关键点提取
        gray_adversarial = cv2.cvtColor(self.adversarial_image, cv2.COLOR_BGR2GRAY)
        faces_adversarial = self.detector(gray_adversarial)
        if len(faces_adversarial) > 0:
            self.adversarial_landmarks = self.predictor(gray_adversarial, faces_adversarial[0])
            # 显示对抗样本识别结果
            self.display_landmarks(self.adversarial_landmarks, self.adversarial_result_label)
            
            # 存储对抗样本关键点数据
            self.all_landmarks_data['adversarial'] = self.extract_landmark_coords(self.adversarial_landmarks)
        else:
            self.adversarial_result_label.setText("未检测到人脸")
            self.all_landmarks_data['adversarial'] = None

        # 应用所有防御方法
        defense_landmarks_list = []
        method_names = []
        
        for widget_info in self.defense_widgets:
            method_name = widget_info['method'].name
            method_names.append(method_name)
            
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
                
                # 获取关键点坐标
                landmark_coords = self.extract_landmark_coords(landmarks)
                defense_landmarks_list.append(landmark_coords)
                self.all_landmarks_data['defenses'][method_name] = landmark_coords
                
                # 计算并显示误差分析 (与原始图像和对抗样本比较)
                self.calculate_error(landmarks, widget_info['error_label'])
            else:
                widget_info['result_label'].setText("未检测到人脸")
                widget_info['error_label'].setText("无法计算误差")
                defense_landmarks_list.append(None)
                self.all_landmarks_data['defenses'][method_name] = None
                
        # 更新误差对比图表
        if self.all_landmarks_data['original'] is not None and self.all_landmarks_data['adversarial'] is not None:
            self.update_error_chart(method_names)
                    
    def showEvent(self, event):
        super().showEvent(event)
        self.load_images()

    def load_images(self):
        # 加载原始图像
        self.original_image = self.main_window.get_captured_image()
        if self.original_image is not None:
            # 显示原始图像
            frame_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.original_image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.original_image_label.size(), Qt.KeepAspectRatio))
        
        # 加载对抗样本
        self.adversarial_image = self.main_window.get_adversarial_image()
        if self.adversarial_image is not None:
            # 显示对抗样本
            frame_rgb = cv2.cvtColor(self.adversarial_image, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.adversarial_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.adversarial_label.size(), Qt.KeepAspectRatio))

    def extract_landmark_coords(self, landmarks):
        """从关键点提取主要特征坐标"""
        coords = []
        
        # 左眼中心
        left_eye = ((landmarks.part(36).x + landmarks.part(39).x) // 2,
                   (landmarks.part(36).y + landmarks.part(39).y) // 2)
        coords.append(left_eye)
        
        # 右眼中心
        right_eye = ((landmarks.part(42).x + landmarks.part(45).x) // 2,
                    (landmarks.part(42).y + landmarks.part(45).y) // 2)
        coords.append(right_eye)
        
        # 鼻尖
        nose = (landmarks.part(30).x, landmarks.part(30).y)
        coords.append(nose)
        
        # 嘴巴中心
        mouth_left = (landmarks.part(48).x, landmarks.part(48).y)
        mouth_right = (landmarks.part(54).x, landmarks.part(54).y)
        mouth_center = ((mouth_left[0] + mouth_right[0]) // 2,
                      (mouth_left[1] + mouth_right[1]) // 2)
        coords.append(mouth_center)
        
        # 下巴
        chin = (landmarks.part(8).x, landmarks.part(8).y)
        coords.append(chin)
        
        return coords

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

    def calculate_error(self, defense_landmarks, label):
        """
        计算处理后的图像与原始图像和对抗样本的误差
        """
        error_text = "误差分析：\n"
        
        if self.original_landmarks is None:
            error_text += "未检测到原始图像人脸\n"
        else:
            # 计算与原始图像的误差
            original_error = self.calculate_landmarks_error(self.original_landmarks, defense_landmarks)
            error_text += f"与原始图像的平均误差: {original_error:.2f}像素\n"
        
        if self.adversarial_landmarks is None:
            error_text += "未检测到对抗样本人脸\n"
        else:
            # 计算与对抗样本的误差
            adversarial_error = self.calculate_landmarks_error(self.adversarial_landmarks, defense_landmarks)
            error_text += f"与对抗样本的平均误差: {adversarial_error:.2f}像素\n"
            
        label.setText(error_text)

    def calculate_landmarks_error(self, landmarks1, landmarks2):
        """计算两组关键点之间的平均误差"""
        total_error = 0
        features = {
            '左眼中心': [(36, 39), (36, 39)],
            '右眼中心': [(42, 45), (42, 45)],
            '鼻尖': [(30, 30), (30, 30)],
            '嘴巴中心': [(48, 54), (48, 54)],
            '下巴': [(8, 8), (8, 8)]
        }
        
        for feature, points in features.items():
            lm1_x = (landmarks1.part(points[0][0]).x + landmarks1.part(points[0][1]).x) // 2
            lm1_y = (landmarks1.part(points[0][0]).y + landmarks1.part(points[0][1]).y) // 2
            lm2_x = (landmarks2.part(points[1][0]).x + landmarks2.part(points[1][1]).x) // 2
            lm2_y = (landmarks2.part(points[1][0]).y + landmarks2.part(points[1][1]).y) // 2
            
            error = np.sqrt((lm1_x - lm2_x)**2 + (lm1_y - lm2_y)**2)
            total_error += error
        
        return total_error / len(features)

    def update_error_chart(self, method_names):
        """更新误差对比图表"""
        # 获取对抗样本和各防御方法的关键点数据
        original_data = self.all_landmarks_data['original']
        adversarial_data = self.all_landmarks_data['adversarial']
        
        defense_data_list = []
        for method in method_names:
            defense_data_list.append(self.all_landmarks_data['defenses'].get(method))
        
        # 更新图表
        self.error_chart.plot_comparison(
            original_data, 
            adversarial_data, 
            defense_data_list, 
            self.feature_names, 
            method_names
        )
        
        # 更新误差详情文本
        error_details = self.generate_error_details()
        self.error_details_label.setText(error_details)

    def generate_error_details(self):
        """生成详细的误差比较文本"""
        if self.all_landmarks_data['original'] is None or self.all_landmarks_data['adversarial'] is None:
            return "无法生成误差详情，缺少原始图像或对抗样本数据"
            
        # 创建表格样式格式
        details = "<html><body style='text-align: center;'><style>"
        details += "table { border-collapse: collapse; width: 90%; margin: 0 auto; }"
        details += "th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }"
        details += "th { background-color: #4CAF50; color: white; }"
        details += "tr:nth-child(even) { background-color: #f2f2f2; }"
        details += "tr:hover { background-color: #ddd; }"
        details += "h3 { text-align: center; }"
        details += "</style>"
        
        # 表格标题
        details += "<h3>各方法与原始图像误差对比表 (单位: 像素)</h3>"
        details += "<table><tr><th>特征点</th><th>对抗样本</th>"
        
        # 添加防御方法列
        for method in self.all_landmarks_data['defenses'].keys():
            details += f"<th>{method}</th>"
        details += "</tr>"
        
        # 计算每个特征点的误差
        for i, feature in enumerate(self.feature_names):
            details += f"<tr><td>{feature}</td>"
            
            # 对抗样本与原始图像的误差
            orig_point = self.all_landmarks_data['original'][i]
            adv_point = self.all_landmarks_data['adversarial'][i]
            adv_error = np.sqrt((orig_point[0] - adv_point[0])**2 + (orig_point[1] - adv_point[1])**2)
            details += f"<td>{adv_error:.1f}</td>"
            
            # 各防御方法与原始图像的误差
            for method, defense_data in self.all_landmarks_data['defenses'].items():
                if defense_data is not None:
                    def_point = defense_data[i]
                    def_error = np.sqrt((orig_point[0] - def_point[0])**2 + (orig_point[1] - def_point[1])**2)
                    
                    # 根据误差大小设置不同颜色
                    if def_error < adv_error * 0.5:  # 误差小于对抗样本一半
                        color = "#4CAF50"  # 绿色 - 优秀
                    elif def_error < adv_error * 0.8:  # 误差小于对抗样本80%
                        color = "#2196F3"  # 蓝色 - 良好
                    elif def_error < adv_error:  # 误差小于对抗样本
                        color = "#FF9800"  # 橙色 - 一般
                    else:  # 误差大于对抗样本
                        color = "#F44336"  # 红色 - 较差
                    
                    details += f"<td style='color: {color};'><b>{def_error:.1f}</b></td>"
                else:
                    details += "<td>N/A</td>"
            details += "</tr>"
        
        # 添加平均误差行
        details += "<tr style='font-weight: bold; background-color: #e7e7e7;'><td>平均误差</td>"
        
        # 计算对抗样本平均误差
        adv_total_error = 0
        for i in range(len(self.feature_names)):
            orig_point = self.all_landmarks_data['original'][i]
            adv_point = self.all_landmarks_data['adversarial'][i]
            adv_error = np.sqrt((orig_point[0] - adv_point[0])**2 + (orig_point[1] - adv_point[1])**2)
            adv_total_error += adv_error
        adv_avg_error = adv_total_error / len(self.feature_names)
        details += f"<td>{adv_avg_error:.1f}</td>"
        
        # 计算各防御方法平均误差
        for method, defense_data in self.all_landmarks_data['defenses'].items():
            if defense_data is not None:
                def_total_error = 0
                for i in range(len(self.feature_names)):
                    orig_point = self.all_landmarks_data['original'][i]
                    def_point = defense_data[i]
                    def_error = np.sqrt((orig_point[0] - def_point[0])**2 + (orig_point[1] - def_point[1])**2)
                    def_total_error += def_error
                def_avg_error = def_total_error / len(self.feature_names)
                
                # 根据平均误差大小设置不同颜色
                if def_avg_error < adv_avg_error * 0.5:
                    color = "#4CAF50"  # 绿色 - 优秀
                elif def_avg_error < adv_avg_error * 0.8:
                    color = "#2196F3"  # 蓝色 - 良好
                elif def_avg_error < adv_avg_error:
                    color = "#FF9800"  # 橙色 - 一般
                else:
                    color = "#F44336"  # 红色 - 较差
                
                details += f"<td style='color: {color};'><b>{def_avg_error:.1f}</b></td>"
            else:
                details += "<td>N/A</td>"
        
        details += "</tr></table>"
        
        # 添加说明
        details += "<p><small>颜色说明：<span style='color: #4CAF50;'>绿色</span> - 优秀 (误差小于对抗样本的50%)，"
        details += "<span style='color: #2196F3;'>蓝色</span> - 良好 (误差小于对抗样本的80%)，"
        details += "<span style='color: #FF9800;'>橙色</span> - 一般 (误差小于对抗样本)，"
        details += "<span style='color: #F44336;'>红色</span> - 较差 (误差大于对抗样本)</small></p>"
        details += "</body></html>"
        
        return details

    def gaussian_defense(self, image):
        """高斯滤波防御方法"""
        return cv2.GaussianBlur(image, (5, 5), 0)

    def median_defense(self, image):
        """中值滤波防御方法"""
        return cv2.medianBlur(image, 5)

    def bilateral_defense(self, image):
        """双边滤波防御方法 - 保留边缘的平滑"""
        return cv2.bilateralFilter(image, 9, 75, 75)
        
    def jpeg_defense(self, image):
        """JPEG压缩防御方法 - 通过压缩去除高频噪声"""
        # 模拟JPEG压缩过程
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # 质量参数，较低的值会增加压缩率
        _, encoded_img = cv2.imencode('.jpg', image, encode_param)
        decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        return decoded_img
        
    def histogram_eq_defense(self, image):
        """直方图均衡化防御方法 - 增强对比度"""
        # 转换到YUV空间以保留色彩
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # 仅对亮度通道进行直方图均衡化
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # 转回BGR空间
        equalized_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return equalized_img
        
    def random_noise_defense(self, image):
        """随机抖动防御方法 - 添加微小随机噪声干扰对抗样本"""
        # 创建弱噪声
        noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
        # 添加到图像上
        noisy_img = cv2.add(image, noise)
        # 确保值在有效范围内
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return noisy_img
        
    def edge_enhance_defense(self, image):
        """边缘增强防御方法 - 强化人脸特征边缘"""
        # 创建锐化核
        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])
        # 应用卷积
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
        
    def color_space_defense(self, image):
        """色彩空间变换防御方法 - 通过色彩空间转换减少对抗干扰"""
        # 转换到LAB空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # 调整L通道
        l, a, b = cv2.split(lab)
        # 应用CLAHE（对比度受限的自适应直方图均衡化）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        # 合并通道
        merged = cv2.merge((cl, a, b))
        # 转回BGR空间
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return enhanced

    
