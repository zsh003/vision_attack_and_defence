import cv2
import dlib
import time
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# 加载dlib的人脸检测器和预测器
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    raise FileNotFoundError(f"File {predictor_path} not found. Please download and place it in the project directory.")

predictor = dlib.shape_predictor(predictor_path)

def get_random_challenge():
    challenges = ["点点头", "摇摇头"]
    return random.choice(challenges)

def put_chinese_text(img, text, position, font_path='simhei.ttf', font_size=30, color=(0, 0, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    except IOError:
        print(f"Font file {font_path} not found. Using default font.")
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

def detect_head_movement(prev_points, curr_points):
    if prev_points is None or curr_points is None:
        return None

    # 计算眼睛中心点的变化
    left_eye_prev = ((prev_points.part(36).x + prev_points.part(39).x) // 2, (prev_points.part(36).y + prev_points.part(39).y) // 2)
    right_eye_prev = ((prev_points.part(42).x + prev_points.part(45).x) // 2, (prev_points.part(42).y + prev_points.part(45).y) // 2)
    mouth_prev = ((prev_points.part(48).x + prev_points.part(54).x) // 2, (prev_points.part(48).y + prev_points.part(54).y) // 2)

    left_eye_curr = ((curr_points.part(36).x + curr_points.part(39).x) // 2, (curr_points.part(36).y + curr_points.part(39).y) // 2)
    right_eye_curr = ((curr_points.part(42).x + curr_points.part(45).x) // 2, (curr_points.part(42).y + curr_points.part(45).y) // 2)
    mouth_curr = ((curr_points.part(48).x + curr_points.part(54).x) // 2, (curr_points.part(48).y + curr_points.part(54).y) // 2)

    eye_diff_x = abs(left_eye_curr[0] - left_eye_prev[0]) + abs(right_eye_curr[0] - right_eye_prev[0])
    eye_diff_y = abs(left_eye_curr[1] - left_eye_prev[1]) + abs(right_eye_curr[1] - right_eye_prev[1])

    mouth_diff_x = abs(mouth_curr[0] - mouth_prev[0])
    mouth_diff_y = abs(mouth_curr[1] - mouth_prev[1])

    head_center_prev = ((left_eye_prev[0] + right_eye_prev[0]) // 2, (left_eye_prev[1] + right_eye_prev[1]) // 2)
    head_center_curr = ((left_eye_curr[0] + right_eye_curr[0]) // 2, (left_eye_curr[1] + right_eye_curr[1]) // 2)

    head_diff_x = abs(head_center_curr[0] - head_center_prev[0])
    head_diff_y = abs(head_center_curr[1] - head_center_prev[1])

    if eye_diff_y > 20 and head_diff_y > 10 and mouth_diff_y < 10:
        return "nod"
    elif head_diff_x > 20 and mouth_diff_x < 10:
        return "shake"

    return None

def main():
    # 尝试打开默认摄像头（索引为0），使用DirectShow后端
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open video device with index 0.")
        # 尝试使用索引1
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Error: Could not open video device with index 1.")
            exit()

    last_challenge_time = 0
    challenge_duration = 3  # 挑战持续时间（秒）
    current_challenge = None
    prev_landmarks = None
    action_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video device.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if current_challenge == "点点头":
                action = detect_head_movement(prev_landmarks, landmarks)
                if action == "nod":
                    action_detected = True
            elif current_challenge == "摇摇头":
                action = detect_head_movement(prev_landmarks, landmarks)
                if action == "shake":
                    action_detected = True

            prev_landmarks = landmarks

        current_time = time.time()

        if current_time - last_challenge_time > challenge_duration:
            current_challenge = get_random_challenge()
            last_challenge_time = current_time
            action_detected = False

        if current_challenge:
            frame = put_chinese_text(frame, f"请{current_challenge}!", (10, 30))

        if action_detected:
            frame = put_chinese_text(frame, "动作已检测到!", (10, 70), color=(0, 255, 0))

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



