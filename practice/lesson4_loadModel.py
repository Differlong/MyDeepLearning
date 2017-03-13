from keras.models import load_model
model = load_model("./mnist_model.h5")
import cv2
img = cv2.imread("thumb.jpg",0)
img = (img.reshape(1,28,28,1)).astype("float32")/255
predict = model.predict_classes(img)
print(predict)
#print(predict.argmax(axis=1))