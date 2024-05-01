from sklearn.metrics import accuracy_score
from random import choice
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
from numpy import load

#: Tải dữ liệu nhúng khuôn mặt từ tệp 'faces_embeddings.npz'.
data = load('faces_embeddings.npz')
trainX,trainy,testX, testy = data['arr_0'], data['arr_1'],data['arr_2'],data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))

# Tạo một đối tượng Normalizer để chuẩn hóa dữ liệu. Trong trường hợp này, norm='l2' chỉ định rằng dữ liệu sẽ được chuẩn hóa bằng cách chia mỗi hàng của mảng cho norm L2 của nó (tổng bình phương các giá trị).
in_encoder = Normalizer(norm='l2')

# Chuẩn hóa dữ liệu huấn luyện bằng cách sử dụng đối tượng Normalizer đã được tạo trước đó.
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# Tạo một đối tượng LabelEncoder để mã hóa nhãn. Đối tượng này sẽ chuyển đổi các nhãn chuỗi thành các nhãn số nguyên.
out_encoder = LabelEncoder()
out_encoder.fit(trainy)

trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# Tạo một mô hình phân loại hỗ trợ (Support Vector Classification) với kernel là tuyến tính và khả năng dự đoán xác suất (probability=True).
model = SVC(kernel='linear', probability=True)

#Huấn luyện mô hình SVC trên dữ liệu huấn luyện đã được chuẩn hóa và nhãn tương ứng.
model.fit(trainX,trainy)

# yhat_train = model.predict(trainX)
# yhat_test = model.predict(testX)

# score_train = accuracy_score(trainy, yhat_train)
# score_test = accuracy_score(testy, yhat_test)
# print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))