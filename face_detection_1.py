import cv2
import dlib
import time
import random

# 加载dlib的HOG+SVM人脸检测器
detector = dlib.get_frontal_face_detector()

def get_random_challenge():
    challenges = ["点点头", "摇摇头"]
    return random.choice(challenges)

def main():
    # 尝试打开默认摄像头（索引为0），使用DirectShow后端
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open video device.")
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
            cv2.putText(frame, f"请{current_challenge}!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



