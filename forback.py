import numpy as np
from backward import backward
from forward import forward
from normalize import normalize

def forback(P,pi,sigma,mu,X):
	T=X.shape[0]
	K=mu.shape[0]
	alpha,_=forward(P,pi,sigma,mu,X)
	beta= backward(P,pi,sigma,mu,X)
	beta.shape
	
	gamma=np.zeros((T,K))
	for i in range(T):
		for j in range(K):
			gamma[i][j]=alpha[i][j] * beta[i][j]
	
	for i in range(T):
		gamma[i,:],_=normalize(gamma[i,:])
	return gamma
    
