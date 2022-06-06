from asyncio.windows_events import NULL
from cgi import print_environ
from turtle import shape
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class Perceptron:

    def __init__(self,MatriceDapprentissage, ResultatsAttendus):
        one = np.ones(MatriceDapprentissage.shape[0])
        self.X= np.c_[one,MatriceDapprentissage]
        self.y = ResultatsAttendus
        self.w =np.random.randn(self.X.shape[1],1)





    def DifAttendueObtenue(self,i):       
        attendue = self.y[i]
        obtenue = (np.dot( self.X[i],self.w ))>0
        return attendue-obtenue
    





    def MettreAJourPoidsLigne(self,i,dif):
        self.w = self.w.T
        self.w = self.w + dif * (self.X)[i].T
        self.w = self.w.T

    def MettreAJourPoids(self):
        for i in range(self.y.shape[0]):
            self.MettreAJourPoidsLigne(i,self.DifAttendueObtenue(i))

    def Perceptron(self,iteration):
        for j in tqdm(range(iteration)):
            self.MettreAJourPoids()
        print("la Matrice des poids est:\n",self.w)

    def AffichePoints(self):
        
        X = self.X
        y = self.y
        for i in range(X.shape[0]):
            if(y[i] == 0):
                plt.scatter(X[i][1],X[i][2],c='red')
            else:
                plt.scatter(X[i][1],X[i][2],c='blue')
    
    def TraceLigneSeparatrice(self):
        x1 = np.linspace(np.min(self.X), np.max(self.X),self.X.shape[1]-1)
        x2 = (-self.w[1] * x1 - self.w[0]) / self.w[2]
        plt.plot(x1,x2,c='green',lw=1)

        
            
    def TraceGraphe(self):
        self.AffichePoints()
        self.TraceLigneSeparatrice()
        plt.show()

    
    
    
    
    
    
    

    
