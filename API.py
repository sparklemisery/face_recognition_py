from flask import Flask, jsonify, request 
import io
from PIL import Image
import cv2 as cv
import numpy as np
from test import process_img


app = Flask(__name__)   

 
@app.route('/', methods = ['GET', 'POST']) 
def home(): 
    if(request.method == 'GET'):   
        data = "hello world"
        return jsonify({'data': data}) 
  


@app.route('/facerecognition', methods=['POST'])
def process_picture():
    #nhận dữ liệu ảnh mã hóa được truyền từ phía java sau đó chuyển đổi thành ảnh để xuử lý
    image_data = request.get_data()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv.imdecode(nparr,cv.IMREAD_COLOR)
    result = process_img(image)
    print('result is:', result)
    return str(result)



if __name__ == '__main__':   
    app.run(debug = True) 

