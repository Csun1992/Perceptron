import numpy as np

def sign(x):
    if np.sign(x)==0:
        return -1
    else:
        return np.sign(x)

class Pla:
    def __init__(self, data, group):
        self.size = np.size(data, 0) 
        self.data = np.concatenate(np.ones(1, self.size), data, axis=1) 
        self.correctGroup = group 
        self.misclassified = list(range(self.size)) 
        self.coeff = np.zeros((np.size(data, 1)+1, 1))
        self.classifiedGroup = [0] * self.size
         
    def updateMisclassified():
        self.misclassified = []
        for i in range(self.size):
           if(self.classifiedGroup[i] != self.label[i]):
              self.misclassified.append(i) 
    
    def pla():
        while not self.misclassified:
            index = self.misclassified[0]
            self.coeff = self.coeff + self.correctGroup[index]*self.data[index, :]
            for i in range(self.size):
                self.classifiedGroup[i] = sign(self.coeff.dot(self.data[i, :]))
                self.updateMisclassified()
