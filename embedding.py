from numpy import load
from numpy import expand_dims
from numpy import asarray
from keras_facenet import FaceNet
import numpy as np

embedder = FaceNet()

# chuyển các ảnh gương mặt thành embedding
def get_embedding(face_img):
    #Chuyển đổi dữ liệu hình ảnh sang kiểu dữ liệu float32
    face_img = face_img.astype('float32')

    #Mở rộng số chiều của mảng face_img bằng cách thêm một chiều mới ở đầu tiên (axis=0). Điều này là để phù hợp với định dạng đầu vào của mô hình nhúng, vì nhiều mô hình yêu cầu đầu vào có một chiều thêm để xác định số lượng mẫu.
    face_img = np.expand_dims(face_img, axis = 0)
    
    # Sử dụng mô hình nhúng (embedder) để tính toán vectơ nhúng của hình ảnh khuôn mặt.
    yhat = embedder.embeddings(face_img)
    return yhat[0]

def get_embedding_array(face_img_array):
    embedding_array = list()
    for face_img in face_img_array:
        embedding_array.append(get_embedding(face_img))
    return embedding_array
    

# data = load('init_face_array.npz')
# trainX, trainy, testX, testy = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']
# print('loaded: ',trainX.shape, trainy.shape, testX.shape, testy.shape)

# newTrainX = list()
# for face_pixel in trainX:
#     embedding = get_embedding(face_pixel)
#     newTrainX.append(embedding)
# newTrainX = asarray(newTrainX)

# newTestX = list()
# for face_pixel in testX:
#     embedding = get_embedding(face_pixel)
#     newTestX.append(embedding)
# newTestX = asarray(newTestX)

# print(newTrainX.shape, newTestX.shape)


# np.savez_compressed('faces_embeddings.npz', newTrainX,trainy,newTestX,testy)

