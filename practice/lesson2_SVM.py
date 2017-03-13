from dataMaker import dataMaker
from sklearn import svm
from matplotlib import pyplot as plt
xTrain,yTrain = dataMaker()
xTest,yTest = dataMaker()

clf = svm.SVC(C=0.1)
clf.fit(xTrain,yTrain)
predict = clf.predict(xTest)
print(1 - (predict - yTest).__abs__().sum()/1000)

for x,y,pre in zip(xTest,yTest,predict):
    if y == pre:
        if y == 1:
            plt.plot(x[0],x[1],"b+")
        else:
            plt.plot(x[0],x[1],"g+")
    else:
        plt.plot(x[0],x[1],"r*")
plt.show()
