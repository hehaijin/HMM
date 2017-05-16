import numpy as np
from normalize import normalize
from scipy.stats import multivariate_normal as mvnorm
from data import markovprocess
from forback import forback
from eta import eta
import hmm
import matplotlib.pyplot as plt


P=np.array([[0.8, 0.1 ,0.1],[0.2, 0.5 , 0.3],[0.3, 0.1, 0.6]])
mu=np.array([1,2,3])
N=100
pi=np.array([0.2,0.3,0.3])
sigma=np.array([0.3, 0.3, 0.3])
X,Z=markovprocess(P,sigma,mu,N) 
gamma=forback(P,pi,sigma,mu,X)
et=eta(P,pi,sigma,mu,X)
K=P.shape[0]
T=X.shape[0]

R=np.zeros(N)
for i in range(N):
	m=max(gamma[i,:])
	for j in range(K):
		if gamma[i,j]==m:
			R[i]=j

plt.subplot(211)
plt.plot(Z)
plt.subplot(212)
plt.plot(R)
plt.show()




#calculate transition matrix P
NN=np.zeros((K,K))
P=np.zeros((K,K))
for j in range(K):
	for k in range(K):
		NN[j,k]=sum(et[:,j,k])
for i in range(K):
	NN[i,:],_=normalize(NN[i,:])
print(NN)



















