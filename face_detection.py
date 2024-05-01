import os
import numpy as np
from numpy import asarray
import cv2 as cv
from PIL import Image
from os.path import isdir
from mtcnn.mtcnn import MTCNN


detector = MTCNN()


# chiết xuất khuôn mặt từ 1 ảnh chỉ có 1 đối tượng , phục vụ cho việc training
def extract_face (filename, required_size=(160,160)):
    # đọc file ảnh
    image = cv.imread(filename)

    # chuyển màu ảnh sang rgb vi cv măc định là bgr
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    #chuyển đổi hình ảnh được đọc bằng OpenCV thành một mảng NumPy
    pixels = asarray(image)

    #xác định vị trí khuôn mặt
    results = detector.detect_faces(pixels)
    if len(results) == 0:
        return None
    x1,y1,width, height = results[0]['box']
    x1,y1 = abs(x1), abs(y1)
    x2,y2 = x1+width, y1+height

    # cắt hình ảnh theo vị trí khuôn mặt
    face = pixels[y1:y2,x1:x2]
    face = cv.cvtColor(face,cv.COLOR_BGR2RGB)
    
    #chuyển đổi một mảng NumPy chứa dữ liệu ảnh thành một đối tượng hình ảnh
    image = Image.fromarray(face)

    #resize ảnh theo kích thước (160,160)
    image = image.resize(required_size)

    cv.imshow('Hình ảnh', face)

    # Chờ phản hồi từ người dùng
    cv.waitKey(0)

    # Đóng cửa sổ hiển thị
    cv.destroyAllWindows()
    
    #chuyển hình ảnh lần nữa thành mảng numpy
    face_array = asarray(image) 
    return face_array


# triết xuất gương mặt phục vụ cho việc nhận diện nhiều người trong ảnh
def extract_face_img(image, required_size=(160,160)):
    faces = list()
    retangles = list()
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    pixels = asarray(image)
    results = detector.detect_faces(pixels)
    print('result : ',len(results))
    for i in range(len(results)):
        x1,y1,width, height = results[i]['box']
        x1,y1 = abs(x1), abs(y1)
        x2,y2 = x1+width, y1+height
        retangles.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2})
        face = pixels[y1:y2,x1:x2]

        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image) 
        faces.append(face_array)
    return faces, retangles

#xác định khuôn mặt của tất cả các ảnh trong thư mục như DE10473
def load_faces(directory):
    faces = list()
    for filename in os.listdir(directory):
        path = directory+'/'+filename
        face = extract_face(path)

        if face is None :
            print("not found face")
            continue
         
        faces.append(face)
    return faces

#xác định khuôn mặt của tất cả các ảnh trong PicTrain hay PicTest
def load_dataset(directory):
    X, y = list(), list()
    for subdir in os.listdir(directory):
        path = directory+'/'+subdir
        if not isdir(path):
            continue
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        X.extend(faces)
        y.extend(labels)
    return asarray(X),asarray(y) 



# trainX, trainy =load_dataset('dataset/PicTrain')
# testX, testy = load_dataset('dataset/PicTest')

# np.savez_compressed('init_face_array.npz',trainX, trainy, testX, testy)
