from face_detection import extract_face,extract_face_img
from embedding import get_embedding,get_embedding_array
from model_recog import model, out_encoder
import json

# xử lý ảnh đầu vào và trả vể mảng nhận diện kèm độ chính xác
def process_img(image):
    results = list()
    imgTest_array,rectangle = extract_face_img(image)

    if len(imgTest_array) == 0:
        return 0
    
    imgTest_array = get_embedding_array(imgTest_array)

    print('the number of images : ',len(imgTest_array))

    for i in range(len(imgTest_array)):
        imgtest = imgTest_array[i].reshape(1,-1)
        yhat_class = model.predict(imgtest)
        yhat_prob = model.predict_proba(imgtest)
        class_index = yhat_class[0]
        class_probablility = yhat_prob[0,class_index] * 100
        predict_name = out_encoder.inverse_transform(yhat_class)
        
        # đặt mức chính xác nhận diện
        if class_probablility > 10:
            results.append({"name":predict_name[0],"probability":class_probablility,"position":rectangle[i]})
        else :
            results.append({"name":"unknown","probability":0.00,"position":rectangle[i]})
    #json.dumps() được sử dụng để chuyển đổi một đối tượng Python thành một chuỗi JSON để lưu trữ hoặc truyền dữ liệu dưới dạng chuỗi JSON.            
    data_json = json.dumps(results)
    
    return data_json
