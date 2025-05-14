import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedWidget
from PyQt5.QtCore import Qt
import PyQt5
import os

dirname = os.path.dirname(PyQt5.__file__)
qt_dir = os.path.join(dirname, 'Qt5', 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_dir

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("机器人视觉控制攻击防御系统")
        self.setGeometry(100, 100, 1200, 800)

        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 创建主布局
        layout = QVBoxLayout(main_widget)

        # 添加标题
        title_label = QLabel("机器人视觉控制攻击防御系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)

        # 创建导航栏
        nav_layout = QHBoxLayout()
        self.recognition_btn = QPushButton("视觉识别")
        self.attack_btn = QPushButton("视觉攻击")
        self.defense_btn = QPushButton("视觉防御")

        for btn in [self.recognition_btn, self.attack_btn, self.defense_btn]:
            btn.setMinimumHeight(40)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 16px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 5px 20px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            nav_layout.addWidget(btn)

        layout.addLayout(nav_layout)

        # 创建堆叠窗口部件
        self.stacked_widget = QStackedWidget()

        from pages.recognition_page import RecognitionPage
        from pages.attack_page import AttackPage
        from pages.defense_page import DefensePage

        self.recognition_page = RecognitionPage(self)
        self.attack_page = AttackPage(self)
        self.defense_page = DefensePage(self)

        self.stacked_widget.addWidget(self.recognition_page)
        self.stacked_widget.addWidget(self.attack_page)
        self.stacked_widget.addWidget(self.defense_page)

        layout.addWidget(self.stacked_widget)

        # 连接按钮信号
        self.recognition_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        self.attack_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        self.defense_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))

        # 设置默认页面
        self.stacked_widget.setCurrentIndex(0)

        self.captured_image = None
        self.adversarial_image = None

    def set_captured_image(self, image):
        self.captured_image = image

    def get_captured_image(self):
        return self.captured_image

    def set_adversarial_image(self, image):
        self.adversarial_image = image

    def get_adversarial_image(self):
        return self.adversarial_image


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())