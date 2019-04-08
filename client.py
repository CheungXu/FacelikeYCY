import cv2
import base64,json
import requests
import datetime

src_img = cv2.imread('ycy.jpg', 1)
dst_img = cv2.imread('ycy2.jpg',1)

src_str = str(base64.b64encode(cv2.imencode('.jpg',src_img)[1]))[2:-1]
dst_str = str(base64.b64encode(cv2.imencode('.jpg',dst_img)[1]))[2:-1]

res = {'src_image': src_str, 'dst_image' : dst_str}
j_res = json.dumps(res)

begin = datetime.datetime.now()
rsp = requests.post('http://0.0.0.0:6006/', json=j_res)
end = datetime.datetime.now()

print(rsp)
print(rsp.text)
print((end - begin).total_seconds())

