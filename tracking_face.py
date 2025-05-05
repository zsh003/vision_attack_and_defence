import sensor, image, time, lcd, gc
from maix import KPU
from modules import ybserial
from robot_Lib import Robot

# --- 硬件初始化 ---
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)  # 320x240
sensor.set_vflip(True)
sensor.set_hmirror(True)
sensor.skip_frames(time=500)
lcd.init()

# --- 外设初始化 ---
ser = ybserial()
bot = Robot(ser)
bot.set_pwm_servo_all(90, 90)  # 初始化舵机

# --- 内存管理配置 ---
MODEL_CONFIG = {
    'face': {
        'path': "/sd/KPU/yolo_face_detect/face_detect_320x240.kmodel",
        'anchor': (0.1075, 0.126875, 0.126875, 0.175, 0.1465625, 0.2246875,
                  0.1953125, 0.25375, 0.2440625, 0.351875, 0.341875, 0.4721875,
                  0.5078125, 0.6696875, 0.8984375, 1.099687, 2.129062, 2.425937),
        'params': {
            'anchor_num': 9,
            'img_w': 320,
            'img_h': 240,
            'net_w': 320,
            'net_h': 240,
            'layer_w': 10,
            'layer_h': 8,
            'threshold': 0.5,
            'nms_value': 0.2,
            'classes': 1
        }
    },
    'landmark': {
        'path': "/sd/KPU/face_detect_with_68landmark/landmark68.kmodel"
    },
    'hand': {
        'path': "/sd/KPU/yolo_hand_detect/hand_detect.kmodel",
        'anchor': (0.8125, 0.4556, 1.1328, 1.2667, 1.8594, 1.4889,
                  1.4844, 2.2000, 2.6484, 2.9333),
        'params': {
            'anchor_num': 5,
            'img_w': 320,
            'img_h': 240,
            'net_w': 320,
            'net_h': 256,
            'layer_w': 10,
            'layer_h': 8,
            'threshold': 0.7,
            'nms_value': 0.3,
            'classes': 1
        }
    }
}

class ModelManager:
    def __init__(self):
        self.models = {}
        self.active_models = set()
        self._init_base_model()

    def _init_base_model(self):
        """优先加载必需的基础模型"""
        try:
            self.load('face')
            self.load('landmark')
        except Exception as e:
            print("Base model init failed:", e)
            self.release_all()
            raise

    def load(self, model_name):
        if model_name in self.active_models:
            return

        #print("Loading ",model_name, " ...")
        gc.collect()
        if model_name in self.models:
            self.models[model_name].deinit()  # 确保先释放原有实例
        self.models[model_name] = KPU()  # 创建新实例
        kpu = self.models[model_name]
        config = MODEL_CONFIG[model_name]

        try:
            kpu.load_kmodel(config['path'])
            if 'anchor' in config:
                kpu.init_yolo2(config['anchor'], **config['params'])
            self.active_models.add(model_name)
            #print(model_name," loaded. Free mem: ",gc.mem_free())
        except Exception as e:
            print("Load error:", e)
            self._handle_load_error(model_name)

    def unload(self, model_name):
        if model_name in self.active_models:
            #print("Unloading ",model_name," ...")
            self.models[model_name].deinit()
            self.active_models.remove(model_name)
            gc.collect()
            #print("After unload ", model_name, " free: ",gc.mem_free())

    def release_all(self):
        for name in list(self.active_models):
            self.unload(name)

    def _handle_memory_error(self, model_name):
        """内存不足时的应急处理"""
        print("Memory low when loading ", model_name)
        if 'hand' in self.active_models:
            self.unload('hand')
        gc.collect()
        self.load(model_name)  # 重试加载

    def _handle_load_error(self, model_name):
        """处理加载失败的情况"""
        print("Critical error loading ",model_name," performing full reset...")
        self.release_all()
        gc.collect()
        self._init_base_model()

model_mgr = ModelManager()

# --- 动作检测参数 ---
GESTURE_THRESHOLD = {
    'nod': {'axis': 'y', 'threshold': 15},
    'shake': {'axis': 'x', 'threshold': 12},
    'hand': {'area': 5000}
}

# --- 核心逻辑 ---
def track_face(img):
    try:
        model_mgr.load('face')
        model_mgr.models['face'].run_with_output(img)
        faces = model_mgr.models['face'].regionlayer_yolo2()

        if not faces:
            return None

        # 取最大的人脸
        face = max(faces, key=lambda f: f[2]*f[3])
        x, y, w, h = face[0], face[1], face[2], face[3]

        # 裁剪人脸区域
        face_img = img.copy(roi=(x, y, w, h)).resize(128, 128)
        face_img.pix_to_ai()

        # 特征点检测
        model_mgr.load('landmark')
        landmarks = model_mgr.models['landmark'].run_with_output(face_img, getlist=True)

        del (face_img)

        return (x, y, w, h, landmarks)

    except MemoryError as e:
        print("Face tracking memory error:", e)
        model_mgr.release_all()
        return None

def detect_hands(img):
    try:
        if gc.mem_free() < 150*1024:  # 150KB以下不检测
            return []

        model_mgr.load('hand')
        hand_img = img.copy().resize(320, 256)
        hand_img.pix_to_ai()
        model_mgr.models['hand'].run_with_output(hand_img)
        del (hand_img)
        return model_mgr.models['hand'].regionlayer_yolo2()
    except Exception as e:
        print("Hand detect error:", e)
        return []

def analyze_gestures(landmarks):
    # 简化的动作分析（实际需根据特征点坐标变化实现）
    nose_y = landmarks[30*2+1]  # 假设第30点是鼻尖
    if nose_y > GESTURE_THRESHOLD['nod']['threshold']:
        return 'nod'
    return None

# --- 主循环 ---
last_gesture = None
gesture_cooldown = 0

while True:
    img = sensor.snapshot()
    gesture = None

    # 人脸跟踪与特征点检测
    face_data = track_face(img)
    if face_data:
        x, y, w, h, landmarks = face_data
        img.draw_rectangle(x, y, w, h, color=(0,255,0))

        # 动作识别
        gesture = analyze_gestures(landmarks)

        # 手部检测（低频检测）
        if time.ticks_ms() % 2000 < 30:  # 每2秒检测一次
            hands = detect_hands(img)
            if hands:
                print("===========================HAND DETECTED=========================")
                gesture = 'hand_detected'
                for h in hands:
                    img.draw_rectangle(h[0], h[1], h[2], h[3], color=(255,0,0))

    # 动作响应逻辑
    if gesture and gesture != last_gesture and gesture_cooldown == 0:
        print("===========================GESTURE DETECTED=========================")
        print("Detected gesture:", gesture)
        bot.set_beep(20)

        # 根据不同动作控制舵机
        if gesture == 'nod':
            # 点头：上下摆动
            for angle in range(90, 120, 5):
                bot.set_pwm_servo_all(90, angle)
                time.sleep_ms(50)
            bot.set_pwm_servo_all(90, 90)

        elif gesture == 'hand_detected':
            # 检测到手：水平扫描
            for angle in range(90, 150, 5):
                bot.set_pwm_servo_all(angle, 90)
                time.sleep_ms(50)
            for angle in range(150, 30, -5):
                bot.set_pwm_servo_all(angle, 90)
                time.sleep_ms(50)
            bot.set_pwm_servo_all(90, 90)

        gesture_cooldown = 30  # 30帧冷却
        last_gesture = gesture

    # 状态显示
    img.draw_string(0, 0, "Mem: %2.1f K" %(gc.mem_free()//1024), color=(255,0,0))
    lcd.display(img)

    # 释放资源
    if face_data is None:
        model_mgr.unload('face')
        model_mgr.unload('landmark')
    if gesture_cooldown > 0:
        gesture_cooldown -= 1

# --- 清理 ---
model_mgr.release_all()
