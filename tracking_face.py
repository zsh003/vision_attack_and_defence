import sensor, image, time, lcd
from maix import KPU
from modules import ybserial
from robot_Lib import Robot
from simplePID import PID

PIDx = (50, 0, 15)
PIDy = (50, 0, 10)
SCALE = 1000.0

PID_x = PID(
    160,
    PIDx[0] / 1.0 / (SCALE),
    PIDx[1] / 1.0 / (SCALE),
    PIDx[2] / 1.0 / (SCALE))

PID_y = PID(
    120,
    PIDy[0] / 1.0 / (SCALE),
    PIDy[1] / 1.0 / (SCALE),
    PIDy[2] / 1.0 / (SCALE))

ACTION_DURATION = 500  # 动作持续时间(ms)
NOD_THRESHOLD = 20     # 点头垂直位移阈值
SHAKE_THRESHOLD = 15   # 摇头水平位移阈值
HAND_SCAN_CYCLES = 3   # 手部检测后的扫描次数
# 动作状态机
class ActionState:
    NONE = 0
    NODDING = 1
    SHAKING_LEFT = 2
    SHAKING_RIGHT = 3
    HAND_SCAN = 4


ser = ybserial()
bot = Robot(ser)
bot.set_beep(50)
bot.set_car_motion(0, 0, 0)
bot.set_pwm_servo(1, 90)
bot.set_pwm_servo(2, 90)


lcd.init()
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time = 100)
clock = time.clock()

anchor = (0.1075, 0.126875, 0.126875, 0.175, 0.1465625, 0.2246875, 0.1953125, 0.25375, 0.2440625, 0.351875, 0.341875, 0.4721875, 0.5078125, 0.6696875, 0.8984375, 1.099687, 2.129062, 2.425937)
kpu = KPU()
kpu.load_kmodel("/sd/KPU/yolo_face_detect/face_detect_320x240.kmodel")
kpu.init_yolo2(anchor, anchor_num=9, img_w=320, img_h=240, net_w=320 , net_h=240 ,layer_w=10 ,layer_h=8, threshold=0.7, nms_value=0.2, classes=1)

lm68_kpu = KPU()# 初始化特征点模型
lm68_kpu.load_kmodel("/sd/KPU/face_detect_with_68landmark/landmark68.kmodel")

# 初始化手部检测模型
hand_anchor = (0.8125, 0.4556, 1.1328, 1.2667, 1.8594, 1.4889, 1.4844, 2.2000, 2.6484, 2.9333)
hand_kpu = KPU()
hand_kpu.load_kmodel("/sd/KPU/yolo_hand_detect/hand_detect.kmodel")
hand_kpu.init_yolo2(hand_anchor, anchor_num=5, img_w=320, img_h=240, net_w=320, net_h=256, layer_w=10, layer_h=8, threshold=0.7, nms_value=0.3, classes=1)

# 状态变量
current_action = ActionState.NONE
action_start_time = 0
original_servo_pos = (90, 90)
action_progress = 0
history_points = deque(maxlen=5)  # 保存最近5帧特征点
state = 0
last_x = 90
last_y = 90
last_s1 = 90
last_s2 = 90

def detect_nod(points):
    """检测点头动作（使用下巴点8和鼻尖点30的垂直距离变化）"""
    if len(history_points) < 5:
        return False

    # 计算下巴点的平均Y坐标变化
    y_changes = [p[8][1] - history_points[i][8][1] for i, p in enumerate(history_points) if i > 0]
    avg_change = sum(y_changes) / len(y_changes)

    # 满足大范围垂直运动且符合点头模式
    if abs(avg_change) > NOD_THRESHOLD:
        # 检查运动方向一致性
        if all(y > 0 for y in y_changes[:3]) and all(y < 0 for y in y_changes[3:]):
            return True
    return False

def detect_shake(points):
    """检测摇头动作（使用鼻梁点27的水平位移）"""
    if len(history_points) < 5:
        return False

    # 计算鼻梁点的平均X坐标变化
    x_changes = [p[27][0] - history_points[i][27][0] for i, p in enumerate(history_points) if i > 0]
    avg_change = sum(x_changes) / len(x_changes)

    if abs(avg_change) > SHAKE_THRESHOLD:
        if avg_change > 0:  # 向右移动
            return "right"
        else:               # 向左移动
            return "left"
    return None

def start_action(action_type):
    global current_action, action_start_time, original_servo_pos
    if current_action != ActionState.NONE:
        return

    current_action = action_type
    action_start_time = time.ticks_ms()
    original_servo_pos = (last_s1, last_s2)

    # 初始化动作参数
    if action_type == ActionState.NODDING:
        bot.set_beep(20)  # 短提示音
    elif action_type in (ActionState.SHAKING_LEFT, ActionState.SHAKING_RIGHT):
        bot.set_beep(30)
    elif action_type == ActionState.HAND_SCAN:
        bot.set_beep(50)

def handle_actions():
    global current_action, action_progress, last_s1, last_s2
    elapsed = time.ticks_diff(time.ticks_ms(), action_start_time)

    if current_action == ActionState.NODDING:
        # 点头动作：上下摆动一次
        if elapsed < ACTION_DURATION//2:
            # 向下移动
            new_pos = original_servo_pos[1] + int(30 * math.sin(math.pi * elapsed/ACTION_DURATION))
            control_servo(original_servo_pos[0], new_pos)
        elif elapsed < ACTION_DURATION:
            # 返回原位
            new_pos = original_servo_pos[1] + int(30 * math.sin(math.pi * elapsed/ACTION_DURATION))
            control_servo(original_servo_pos[0], new_pos)
        else:
            current_action = ActionState.NONE

    elif current_action == ActionState.SHAKING_LEFT:
        # 左摇头动作：左右摆动一次
        angle = 90 + int(30 * math.sin(2*math.pi * elapsed/ACTION_DURATION))
        control_servo(angle, original_servo_pos[1])
        if elapsed > ACTION_DURATION:
            current_action = ActionState.NONE

    elif current_action == ActionState.SHAKING_RIGHT:
        # 右摇头动作：反向摆动
        angle = 90 - int(30 * math.sin(2*math.pi * elapsed/ACTION_DURATION))
        control_servo(angle, original_servo_pos[1])
        if elapsed > ACTION_DURATION:
            current_action = ActionState.NONE

    elif current_action == ActionState.HAND_SCAN:
        # 手部扫描：水平往复运动
        cycles = elapsed // (ACTION_DURATION*2)
        if cycles >= HAND_SCAN_CYCLES:
            current_action = ActionState.NONE
            control_servo(*original_servo_pos)
        else:
            phase = elapsed % (ACTION_DURATION*2)
            angle = 90 + 30 * math.sin(math.pi * phase/ACTION_DURATION)
            control_servo(int(angle), original_servo_pos[1])


def control_servo(s1, s2):
    global last_s1, last_s2
    if last_s1 != int(s1) or last_s2 != int(s2):
        last_s1 = int(s1)
        last_s2 = int(s2)
        bot.set_pwm_servo_all(last_s1, last_s2)

hand_detect_interval = 5
hand_counter = 0

while True:
    clock.tick()
    img = sensor.snapshot()
    kpu.run_with_output(img)
    dect = kpu.regionlayer_yolo2()
    fps = clock.fps()

    # 手部检测（降低检测频率）
    hand_counter = (hand_counter + 1) % hand_detect_interval
    if hand_counter == 0 and current_action == ActionState.NONE:
        od_img = img.copy().resize(320, 256)
        od_img.pix_to_ai()
        hand_kpu.run_with_output(od_img)
        hand_dect = hand_kpu.regionlayer_yolo2()
        if len(hand_dect) > 0:
            start_action(ActionState.HAND_SCAN)
    if current_action != ActionState.NONE:
        handle_actions()
    else:
        if len(dect) > 0:
            # 人脸追踪逻辑
            #print("dect:",dect)
            for face in dect :
                pass
            img.draw_rectangle(face[0],face[1],face[2],face[3], color=(0, 255, 0))
            #print("face:", face)
            face_x = face[0] + face[2]//2
            face_y = face[1] + face[3]//2
            #print("face_xy", face_x, face_y)
            value_x = PID_x.incremental(face_x)
            value_y = -PID_y.incremental(face_y)
            if -1 < value_x < 1: value_x = 0
            if -1 < value_y < 1: value_y = 0
            last_x = last_x + value_x
            last_y = last_y + value_y
            control_servo(last_x, last_y)
            print("value:", last_x, last_y)

            # 特征点检测
            face = dect[0]
            x1, y1, cut_w, cut_h = extend_box(face[0], face[1], face[2], face[3], RATIO)
            face_cut = img.cut(x1, y1, cut_w, cut_h)
            face_cut_128 = face_cut.resize(128, 128)
            face_cut_128.pix_to_ai()
            out = lm68_kpu.run_with_output(face_cut_128, getlist=True)

            # 解析68个特征点
            points = []
            for j in range(68):
                # 计算x坐标
                x_ratio = KPU.sigmoid(out[2*j])
                pixel_x = int(x_ratio * cut_w) + x1

                # 计算y坐标
                y_ratio = KPU.sigmoid(out[2*j+1])
                pixel_y = int(y_ratio * cut_h) + y1

                # 坐标边界约束
                pixel_x = max(0, min(319, pixel_x))
                pixel_y = max(0, min(239, pixel_y))

                points.append( (pixel_x, pixel_y) )

            history_points.append(points)

            # 动作检测
            if detect_nod(points):
                start_action(ActionState.NODDING)
            else:
                shake_dir = detect_shake(points)
                if shake_dir == "left":
                    start_action(ActionState.SHAKING_LEFT)
                elif shake_dir == "right":
                    start_action(ActionState.SHAKING_RIGHT)

            # 绘制特征点
            for x, y in points:
                img.draw_circle(x, y, 2, color=(0,0,255), fill=True)
    img.draw_string(0, 0, "%2.1ffps" %(fps), color=(0, 60, 255), scale=2.0)
    lcd.display(img)

kpu.deinit()

