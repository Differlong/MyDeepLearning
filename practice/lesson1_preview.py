from dataMaker import dataMaker
from keras.models import Sequential,Model
from keras.layers import Dense,Input,Activation,Dropout,Convolution1D,Layer,Merge,Lambda,RepeatVector,Flatten
import numpy as np
from keras import backend as K
from keras.utils.np_utils import to_categorical


xTrain,y_Train = dataMaker()
xTest,y_Test = dataMaker()
yTrain = to_categorical(y_Train,2)
yTest = to_categorical(y_Test,2)



batch_size = 32
nb_epoch = 100

# #Model 1
left_branch = Sequential()
left_branch.add(Dense(16,input_dim=2))
left_branch.add(Dropout(0.1))

right_branch = Sequential()
right_branch.add(Dense(16,input_dim=2))
right_branch.add(Dropout(0.1))

merged = Merge([left_branch,right_branch],mode="mul")
model = Sequential()
model.add(merged)
model.add(Dense(8))
model.add(Dense(2,activation="softmax"))




model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit([xTrain,xTrain], yTrain,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=([xTest,xTest], yTest))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    pred = model.predict_classes([xTest,xTest],verbose=0)
    for i in range(1000):
        if pred[i] == y_Test[i]:
            if pred[i] == 0:
                plt.plot(xTest[i][0],xTest[i][1],"b+")
            else:
                plt.plot(xTest[i][0],xTest[i][1],"g+")
        else:
            plt.plot(xTest[i][0],xTest[i][1],"r*")
    plt.show()

# exam = np.array([[0.5,1.5],[0.5,0.5],[0.1,-0.5],[1,1],[0,0],[0.1,0.1],[2,2]])
# predict = model.predict([exam,exam])
# print("\n\Predict:\n",exam,"\n",predict)