import numpy as np
import matplotlib.pyplot as plt

def classification(x: np.array):
    assert (len(x) == 2)
    d = np.dot(x,x) - 2
    if d>0:
        return 1
    else:
        return 0
    #d = x[0] ** 2 + x[1] ** 2 - 1
    # if d > 0.25:
    #     return 1
    # if d >= 0:
    #     p = abs(d) + 0.5
    #     intP = 1 // p + 1
    #     if np.random.randint(intP):
    #         return 1
    #     else:
    #         return 0
    # if d < -0.36:
    #     return 0
    # else:
    #     p = abs(d) + 0.5
    #     intP = 1 // p + 1
    #     if np.random.randint(intP):
    #         return 0
    #     else:
    #         return 1
def dataMaker():
    """
    data dim = (1000,2)
    label:0,1
    classification function x2+y2 = 1

    return:data,label
    """
    xData = 4 * (np.random.random((1000,2)) - 0.5)
    yData = np.zeros(1000,dtype=np.bool)
    for i in range(len(xData)):
        yData[i] = classification(xData[i])
    return xData,yData

if __name__ == "__main__":
    xData,yData = dataMaker()
    for i in range(1000):
        if yData[i]:
            plt.plot(xData[i][0],xData[i][1],"g+")
        else:
            plt.plot(xData[i][0],xData[i][1],"b+")

    theta = np.linspace(0,2*np.pi,250)
    x = np.sqrt(2)*np.cos(theta)
    y = np.sqrt(2)*np.sin(theta)
    plt.plot(x,y,"b")
    plt.show()





