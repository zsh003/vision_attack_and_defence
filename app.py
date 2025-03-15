from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)

def process_image(image_data):
    # 将base64图像数据转换为numpy数组
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.array(image)
    return image_array

@app.route('/api/recognize', methods=['POST'])
def recognize():
    try:
        data = request.json
        image_data = data['image']
        image = process_image(image_data)
        
        # TODO: 实现物体识别逻辑
        x, y = np.random.randint(0, 100, 2)
        
        return jsonify({
            'success': True,
            'coordinates': {'x': int(x), 'y': int(y)}
        })
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
        
        # TODO: 实现对抗样本生成逻辑
        # 简单的图像扰动
        noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        adversarial_image = cv2.add(image, noise)
        
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