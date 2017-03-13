import cv2
import numpy as np
from keras.models import load_model


def imgPreprocess(file):
    '''
    用手机在白字上拍的手写字母
    :param file:
    :return: 1*28*28*1 thumb，tf的格式
    '''
    img = cv2.imread(file,0)
    # row = img.shape[0]
    # col = img.shape[1]
    # r = min(row,col)
    # img = img[(row-r)//2:(row+r)//2][(col-r)//2:(col+r)//2]
    #我自己在这里做的就是二值化，不过好笨就是了
    threshold = 64

    img[img>threshold] = 255
    img[img<=threshold] = 0
    thumb = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)  # 采用了像素相关重采样来做，终于没有出问题，resize可以了！
    #这里最好是做PCA分析
    #这个可以放到代码库里面去了
    thumb[thumb == 250] = 255
    thumb[thumb < 250] = 0
    cv2.imshow("",thumb)
    cv2.imwrite("thumb.jpg",thumb)
    cv2.waitKey()
    thumb = (thumb.reshape(1,28,28,1)).astype("float32")/255
    return thumb
if __name__ == "__main__":
    thumb = imgPreprocess("img333.jpg")
    model = load_model("mnist_model.h5")
    predict = model.predict(thumb)
    print(predict)
    print(predict.argmax(axis=1))

