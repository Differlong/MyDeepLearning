#Mnist
import numpy as np
np.random.seed(2048)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D,MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras import callbacks


#做一个准确率达到一定程度自动停下来的
#remote = callbacks.TensorBoard(log_dir="./logs",histogram_freq=0)
earlyStopping = callbacks.EarlyStopping(monitor="val_loss",patience=2,mode="auto")
batch_size = 128
nb_classes = 10
nb_epoch = 50

img_rows,img_cols = 28,28
nb_filters = 32
pool_size = (2,2)
kernel_size = (5,5)

(X_train,y_train),(X_test,y_test) = mnist.load_data("E:\keas数据集/mnist.pkl.gz")



#需要对数据进行处理，适应后端

if K.image_dim_ordering() == "th":
    X_train = X_train.reshape(X_train.shape[0],1,img_rows,img_cols)
    X_test = X_test.reshape(X_test.shape[0],1,img_rows,img_cols)
    input_shape = (1,img_rows,img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train /=255
X_test /= 255


Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#数据预处理完毕，现在就是需要构建神经网络了

model = Sequential()
model.add(Convolution2D(nb_filters,kernel_size[0],kernel_size[1],border_mode="valid",input_shape=input_shape))
model.add(Activation("relu"))
model.add(Convolution2D(nb_filters,kernel_size[0],kernel_size[1]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy",optimizer="adadelta",metrics=["accuracy"])
model.fit(X_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(X_test,Y_test),callbacks=[earlyStopping])

model.save("./mnist_model.py")















