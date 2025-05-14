import sensor, image, time, lcd
from maix import KPU
from modules import ybserial
from robot_Lib import Robot
from simplePID import PID
import gc

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

lm68_kpu = KPU()
lm68_kpu.load_kmodel("/sd/KPU/face_detect_with_68landmark/landmark68.kmodel")

state = 0
last_x = 90
last_y = 90
last_s1 = 90
last_s2 = 90

# 动作检测参数
NOD_THRESHOLD = 0.10  # 点头阈值（归一化值的变化量）
SHAKE_THRESHOLD = 0.15  # 摇头阈值（归一化值的变化量）
NOD_COOLDOWN = 15     # 点头检测冷却时间（帧数）
last_gesture = None
gesture_cooldown = 0
# 鼻尖位置追踪
nose_y_history = []
nose_x_history = []
HISTORY_SIZE = 8  # 历史记录长度

def control_servo(s1, s2):
    global last_s1, last_s2
    if last_s1 != int(s1) or last_s2 != int(s2):
        last_s1 = int(s1)
        last_s2 = int(s2)
        bot.set_pwm_servo_all(last_s1, last_s2)

def servo_nod(s1, s2):
    time.sleep_ms(50)
    bot.set_pwm_servo_all(s1, s2)
    time.sleep_ms(50)
    angle = 30
    for i in range(2):
        bot.set_pwm_servo_all(s1, s2 + angle)
        time.sleep_ms(300)
        bot.set_pwm_servo_all(s1, s2)
        time.sleep_ms(500)
    print("Nod action execute!")

def servo_left_swerve(s1, s2):
    time.sleep_ms(50)
    bot.set_pwm_servo_all(s1, s2)
    time.sleep_ms(50)
    angle = 30
    for i in range(2):
        bot.set_pwm_servo_all(s1 - angle, s2)
        time.sleep_ms(300)
        bot.set_pwm_servo_all(s1, s2)
        time.sleep_ms(500)
    print("Left swerve action execute!")

def servo_right_swerve(s1, s2):
    time.sleep_ms(50)
    bot.set_pwm_servo_all(s1, s2)
    time.sleep_ms(50)
    angle = 30
    for i in range(2):
        bot.set_pwm_servo_all(s1 + angle, s2)
        time.sleep_ms(300)
        bot.set_pwm_servo_all(s1, s2)
        time.sleep_ms(500)
    print("Right swerve action execute!")


def extend_box(x, y, w, h, scale):
    x1_t = x - scale*w
    x2_t = x + w + scale*w
    y1_t = y - scale*h
    y2_t = y + h + scale*h
    x1 = int(x1_t) if x1_t>1 else 1
    x2 = int(x2_t) if x2_t<320 else 319
    y1 = int(y1_t) if y1_t>1 else 1
    y2 = int(y2_t) if y2_t<240 else 239
    cut_img_w = x2-x1+1
    cut_img_h = y2-y1+1
    return x1, y1, cut_img_w, cut_img_h

def analyze_gestures(landmarks):
    global nose_y_history, nose_x_history
    # 检测点头动作
    # 第30点为鼻尖
    nose_y = landmarks[30*2+1]  # 归一化值（上下方向）
    nose_x = landmarks[30*2]    # 归一化值（左右方向）
    print("nose_y:", nose_y, "nose_x:", nose_x)

    # 更新鼻尖位置历史
    nose_y_history.append(nose_y)
    nose_x_history.append(nose_x)

    if len(nose_y_history) > HISTORY_SIZE:
        nose_y_history.pop(0)
    if len(nose_x_history) > HISTORY_SIZE:
        nose_x_history.pop(0)

    # 至少需要有足够的历史数据才能判断动作
    if len(nose_y_history) < 4 or len(nose_x_history) < 4:
        return None

    # 检测点头动作（上下运动）
    nod_result = detect_nod()
    if nod_result:
        return nod_result

    # 检测摇头动作（左右运动）
    shake_result = detect_shake()
    if shake_result:
        return shake_result

    return None

def detect_nod():
    global nose_y_history, nose_x_history
    # 对数据进行平滑处理
    smoothed_y = []
    for i in range(1, len(nose_y_history)):
        # 简单的移动平均
        avg = (nose_y_history[i] + nose_y_history[i-1]) / 2
        smoothed_y.append(avg)

    # 计算平滑后数据的变化量
    y_changes = [smoothed_y[i] - smoothed_y[i-1] for i in range(1, len(smoothed_y))]

    # 检测点头模式：y值先明显增加（低头）然后明显减少（抬头）
    # 先找到最大增加和最大减少
    max_increase = max(y_changes) if y_changes else 0
    min_decrease = min(y_changes) if y_changes else 0

    # 判断是否满足点头条件
    is_nod = (max_increase > NOD_THRESHOLD and min_decrease < -NOD_THRESHOLD and
             abs(max_increase) + abs(min_decrease) > NOD_THRESHOLD * 3)

    if is_nod:
        print("Nod detected! Changes:", y_changes)
        print("Max increase:", max_increase, "Min decrease:", min_decrease)
        nose_x_history = [] # 清空历史记录 以增加下一次检测准确性
        nose_y_history = []
        return 'nod'

    return None

def detect_shake():
    global nose_y_history, nose_x_history
    # 对数据进行平滑处理
    smoothed_x = []
    for i in range(1, len(nose_x_history)):
        # 简单的移动平均
        avg = (nose_x_history[i] + nose_x_history[i-1]) / 2
        smoothed_x.append(avg)

    # 计算平滑后数据的变化量
    x_changes = [smoothed_x[i] - smoothed_x[i-1] for i in range(1, len(smoothed_x))]

    if not x_changes:
        return None

    # 检测是否存在足够大的水平移动
    max_right = max(x_changes)  # 正值，向右移动
    max_left = min(x_changes)   # 负值，向左移动

    # 首先判断幅度是否足够大
    if abs(max_left) < SHAKE_THRESHOLD or abs(max_right) < SHAKE_THRESHOLD:
        return None  # 变化不够明显

    # 计算整体趋势，判断主要运动方向
    # 将变化分为三类：明显向左、明显向右、微小变化
    left_moves = [x for x in x_changes if x < -SHAKE_THRESHOLD * 0.5]
    right_moves = [x for x in x_changes if x > SHAKE_THRESHOLD * 0.5]

    # 检查是否有足够的相对方向变化
    if not left_moves or not right_moves:
        return None

    # 计算初始方向：观察前半部分数据的趋势
    first_half = x_changes[:len(x_changes)//2]
    first_left = [x for x in first_half if x < -SHAKE_THRESHOLD * 0.5]
    first_right = [x for x in first_half if x > SHAKE_THRESHOLD * 0.5]

    # 通过首个明显移动的方向判断
    initial_move_idx = -1
    initial_direction = "none"

    for i, change in enumerate(x_changes):
        if abs(change) > SHAKE_THRESHOLD * 0.7:
            initial_move_idx = i
            initial_direction = "left" if change > 0 else "right"
            break

    if initial_direction == "none":
        return None

    # 尝试检测方向变化次数
    direction_changes = 0
    last_dir = initial_direction
    for change in x_changes[initial_move_idx+1:]:
        if abs(change) < SHAKE_THRESHOLD * 0.5:  # 忽略微小变化
            continue

        current_dir = "left" if change > 0 else "right"
        if current_dir != last_dir:
            direction_changes += 1
            last_dir = current_dir

    # 判断是否完成摇头动作（至少有一次方向变化）
    if direction_changes < 1:
        return None

    # 分析最明显的幅度变化
    abs_max_left = abs(max_left)
    abs_max_right = abs(max_right)

    # 计算总体偏移量（所有左变化和右变化的累加）
    sum_left = sum(left_moves)
    sum_right = sum(right_moves)

    # 调试输出
    print("Initial direction:", initial_direction)
    print("Direction changes:", direction_changes)
    print("Total left movement:", sum_left)
    print("Total right movement:", sum_right)

    # 根据初始方向和累积变化判断摇头类型
    if initial_direction == "left" and abs_max_left > abs_max_right * 0.7:
        print("Left shake detected! Changes:", x_changes)
        print("Max left:", max_left, "Max right:", max_right)
        nose_x_history = [] # 清空历史记录 以增加下一次检测准确性
        nose_y_history = []
        return 'left_shake'

    if initial_direction == "right" and abs_max_right > abs_max_left * 0.7:
        print("Right shake detected! Changes:", x_changes)
        print("Max right:", max_right, "Max left:", max_left)
        nose_x_history = [] # 清空历史记录 以增加下一次检测准确性
        nose_y_history = []
        return 'right_shake'

    return None

try:
    while True:
        gc.collect()
        clock.tick()
        img = sensor.snapshot()
        kpu.run_with_output(img)
        dect = kpu.regionlayer_yolo2()
        fps = clock.fps()
        gesture = None

        if len(dect) > 0:
            for face in dect :
                pass
            x1, y1, cut_img_w, cut_img_h = extend_box(face[0], face[1], face[2], face[3], scale=0.08) # 扩大人脸框
            face_cut = img.cut(x1, y1, cut_img_w, cut_img_h)
            a = img.draw_rectangle(face[0],face[1],face[2],face[3], color=(0, 255, 0))
            face_cut_128 = face_cut.resize(128, 128)
            face_cut_128.pix_to_ai()
            out = lm68_kpu.run_with_output(face_cut_128, getlist=True)
            face_x = face[0] + face[2]//2
            face_y = face[1] + face[3]//2
            value_x = PID_x.incremental(face_x)
            value_y = -PID_y.incremental(face_y)
            if -1 < value_x < 1: value_x = 0
            if -1 < value_y < 1: value_y = 0
            last_x = last_x + value_x
            last_y = last_y + value_y
            control_servo(last_x, last_y)
            #print("value:", last_x, last_y)

            # 分析动作手势
            gesture = analyze_gestures(out)

            for j in range(68):
                x = int(KPU.sigmoid(out[2 * j])*cut_img_w + x1)
                y = int(KPU.sigmoid(out[2 * j + 1])*cut_img_h + y1)
                #a = img.draw_cross(x, y, size=1, color=(0, 0, 255))
                a = img.draw_circle(x, y, 2, color=(0, 0, 255), fill=True)
            del (face_cut_128)
            del (face_cut)

            # 根据检测到的手势执行动作
            if gesture and gesture != last_gesture and gesture_cooldown == 0:
                print("===========================GESTURE DETECTED=========================")
                print("Detected gesture:", gesture)
                bot.set_beep(20)

                if gesture == 'nod':
                    servo_nod(last_x, last_y)
                elif gesture == 'left_shake':
                    servo_left_swerve(last_x, last_y)
                elif gesture == 'right_shake':
                    servo_right_swerve(last_x, last_y)

                # 设置冷却时间，避免连续触发
                gesture_cooldown = NOD_COOLDOWN
            last_gesture = gesture
            gesture = None

        img.draw_string(0, 0, "%2.1ffps" %(fps), color=(0, 60, 255), scale=2.0)
        if gesture:
            img.draw_string(0, 20, "Gesture: %s" %(gesture), color=(255, 0, 0), scale=2.0)
        lcd.display(img)

        # 冷却时间递减
        if gesture_cooldown > 0:
            gesture_cooldown -= 1

        gc.collect()
except:
    kpu.deinit()
    lm68_kpu.deinit()

