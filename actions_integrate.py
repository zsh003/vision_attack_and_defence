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
    angle = 20
    for i in range(2):
        bot.set_pwm_servo_all(s1, s2 + angle)
        time.sleep_ms(300)
        bot.set_pwm_servo_all(s1, s2 - angle)
        time.sleep_ms(500)

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


while True:
    gc.collect()
    clock.tick()
    img = sensor.snapshot()
    kpu.run_with_output(img)
    dect = kpu.regionlayer_yolo2()
    fps = clock.fps()
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
        print("value:", last_x, last_y)
        for j in range(68):
            x = int(KPU.sigmoid(out[2 * j])*cut_img_w + x1)
            y = int(KPU.sigmoid(out[2 * j + 1])*cut_img_h + y1)
            #a = img.draw_cross(x, y, size=1, color=(0, 0, 255))
            a = img.draw_circle(x, y, 2, color=(0, 0, 255), fill=True)
        del (face_cut_128)
        del (face_cut)

        servo_nod(last_x, last_y)

    img.draw_string(0, 0, "%2.1ffps" %(fps), color=(0, 60, 255), scale=2.0)
    lcd.display(img)
    gc.collect()

kpu.deinit()
lm68_kpu.deinit()

