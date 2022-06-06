import numpy as np
import PerceptronClasse as pi



with open("baseApprentissage.csv") as monFichier:
    array = np.genfromtxt(monFichier,delimiter=";",skip_header=1)
print()

X = array[:,[1,2]]
y = array[:,[3]]

PerceptronOU = pi.Perceptron(X,y)
PerceptronOU.Perceptron(1000)
PerceptronOU.TraceGraphe()







'''
#--------------------------------------------------#

PerceptronOU = pi.PaireImpaire(M_OU,R_OU)
print("La perceptron pour le OU")
PerceptronOU.Perceptron(10000)
PerceptronOU.ligneDeDecision()
'''

'''

#--------------------------------------------------#
M_ET = np.array([[0,0],[0,1],[1,0],[1,1]])
R_ET = np.array([0,0,0,1])
PerceptronET = pi.PaireImpaire(M_ET,R_ET)
print("La perceptron pour le ET")
PerceptronET.Perceptron(10000)
PerceptronET.ligneDeDecision()

'''
#-------------------------------------------------
'''
X, y = pi.make_blobs(n_samples=100, n_features = 2)
Perceptron = pi.PaireImpaire(X,y)
print("La perceptron pour")
Perceptron.Perceptron(5000)
Perceptron.ligneDeDecision()

'''

