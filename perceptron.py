import numpy as np
import sys

def sign(x):
    if np.sign(x)==0:
        return -1
    else:
        return int(np.sign(x))

def findLine(point1, point2):
    slope = (point2[1]-point1[1])/float(point2[0]-point1[0])
    intercept = point2[1] - slope*point2[0]
    return slope, intercept


def classify(data, slope, intercept):
    group = []
    size = np.size(data, 0)
    for i in range(size):
        group.append(sign(data[i,1]-slope*data[i,0]-intercept))
    return group


class Pla:
    def __init__(self, data, group):
        self.size = np.size(data, 0) 
        self.data = np.concatenate((np.ones((self.size,1)), data), axis=1) 
        self.correctGroup = group 
        self.misclassified = list(range(self.size)) 
        self.coeff = np.zeros((np.size(data, 1)+1, 1))
        self.classifiedGroup = [0] * self.size
        self.repeat = 0

    def updateMisclassified(self):
        self.misclassified = []
        for i in range(self.size):
           if(self.classifiedGroup[i] != self.correctGroup[i]):
              self.misclassified.append(i) 
    
    def pla(self):
        while self.misclassified != []:
            self.repeat += 1
            index = self.misclassified[0]
            self.coeff = self.coeff + (self.correctGroup[index]*self.data[index, :]).reshape(-1, 1)
            for i in range(self.size):
                self.classifiedGroup[i] = sign(self.coeff.T.dot(self.data[i, :].reshape(-1, 1)))
                self.updateMisclassified()
        return repeat

    def getCoeff(self):
        return self.coeff

    def getRepetition(self):
        return self.repeat

if __name__ == "__main__":
    data = np.random.uniform(-1, 1, 20).reshape(10, 2)
    x1 = data[0, :]
    x2 = data[1, :]
    slope, intercept = findLine(x1, x2)
    group = classify(data, slope, intercept)

    perceptron = Pla(data, group)
    repeat = perceptron.pla()
