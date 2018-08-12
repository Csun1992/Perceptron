import numpy as np

class Pla:
    def __init__(self, data, group):
        self.data = data
        self.correctGroup = group 
        self.size = np.size(data, 0) 
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
             
