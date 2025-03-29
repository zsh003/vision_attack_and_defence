import cv2
import dlib
import time
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 加载dlib的HOG+SVM人脸检测器
detector = dlib.get_frontal_face_detector()

def get_random_challenge():
    challenges = ["点点头", "摇摇头"]
    return random.choice(challenges)

def put_chinese_text(img, text, position, font_path='simhei.ttf', font_size=30, color=(0, 0, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    draw.text(position, text, font=font, fill=color)
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

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

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video device.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        current_time = time.time()

        if current_time - last_challenge_time > challenge_duration:
            current_challenge = get_random_challenge()
            last_challenge_time = current_time

        if current_challenge:
            frame = put_chinese_text(frame, f"请{current_challenge}!", (10, 30))

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



