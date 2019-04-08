from flask import request, Flask, jsonify
import json, base64
import numpy as np
import cv2
from compare import Facelike

app = Flask(__name__)

facelike = None

def get_image(img):
    img = base64.b64decode(img)
    img = np.fromstring(img, np.uint8)
    img = cv2.imdecode(img, cv2.COLOR_RGB2BGR)
    return img

@app.before_first_request
def first_request():
    global facelike 
    facelike = Facelike()

@app.route('/', methods=['POST'])
def get_frame():
    res = request.json
    img_dict = eval(res)
    src_frame = get_image(img_dict["src_image"])
    dst_frame = get_image(img_dict["dst_image"])
    _, score = facelike.face_similar(src_frame, dst_frame)
    
    res = {'score': str(score)}
    return jsonify(res)

if __name__ == '__main__':
    app.run(host=('0.0.0.0'),port=6006)
