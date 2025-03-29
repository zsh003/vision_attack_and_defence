from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import io
from PIL import Image
import dlib

app = Flask(__name__)

# 加载dlib的人脸检测器和预测器
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

def process_image(image_data):
    # 将base64图像数据转换为numpy数组
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.array(image)
    return image_array

def extract_face_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) > 0:
        landmarks = predictor(gray, faces[0])
        return landmarks
    return None

@app.route('/api/recognize', methods=['POST'])
def recognize():
    try:
        data = request.json
        image_data = data['image']
        image = process_image(image_data)
        
        landmarks = extract_face_landmarks(image)
        if landmarks is not None:
            # 提取关键点坐标
            result = {
                'success': True,
                'landmarks': {
                    'left_eye': {
                        'x': (landmarks.part(36).x + landmarks.part(39).x) // 2,
                        'y': (landmarks.part(36).y + landmarks.part(39).y) // 2
                    },
                    'right_eye': {
                        'x': (landmarks.part(42).x + landmarks.part(45).x) // 2,
                        'y': (landmarks.part(42).y + landmarks.part(45).y) // 2
                    },
                    'nose': {
                        'x': landmarks.part(30).x,
                        'y': landmarks.part(30).y
                    },
                    'mouth': {
                        'x': (landmarks.part(48).x + landmarks.part(54).x) // 2,
                        'y': (landmarks.part(48).y + landmarks.part(54).y) // 2
                    },
                    'chin': {
                        'x': landmarks.part(8).x,
                        'y': landmarks.part(8).y
                    }
                }
            }
        else:
            result = {
                'success': False,
                'error': 'No face detected in the image'
            }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/generate_adversarial', methods=['POST'])
def generate_adversarial():
    try:
        data = request.json
        image_data = data['image']
        image = process_image(image_data)
        
        # 获取人脸关键点
        landmarks = extract_face_landmarks(image)
        if landmarks is None:
            return jsonify({
                'success': False,
                'error': 'No face detected in the image'
            })
        
        # 生成对抗样本（示例：在关键点周围添加扰动）
        adversarial_image = image.copy()
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # 在关键点周围添加随机扰动
            noise = np.random.normal(0, 5, 2).astype(np.int32)
            cv2.circle(adversarial_image, (x + noise[0], y + noise[1]), 2, (0, 0, 255), -1)
        
        # 将处理后的图像转换为base64
        pil_image = Image.fromarray(adversarial_image)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_str}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/defend', methods=['POST'])
def defend():
    try:
        data = request.json
        image_data = data['image']
        method = data['method']
        image = process_image(image_data)
        
        # 应用防御方法
        if method == 'gaussian':
            processed = cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            processed = cv2.medianBlur(image, 5)
        elif method == 'bilateral':
            processed = cv2.bilateralFilter(image, 9, 75, 75)
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid defense method'
            })
        
        # 将处理后的图像转换为base64
        pil_image = Image.fromarray(processed)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_str}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000) 