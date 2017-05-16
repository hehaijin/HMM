import numpy as np
from backward import backward
from forward import forward
from forback import forback
from scipy.stats import multivariate_normal as mvnorm



def eta(P,pi,sigma,mu,X):
	
	T=X.shape[0]
	K=mu.shape[0]
	alpha,_=forward(P,pi,sigma,mu,X)
	beta= backward(P,pi,sigma,mu,X)
	gamma=forback(P,pi,sigma,mu,X)
	R=np.zeros(T)
	for i in range(T):
		m=max(gamma[i,:])
		for j in range(K):
			if gamma[i,j]==m:
				R[i]=j
	
	
	et=np.zeros((T,K,K))
	for t in range(T-1):
		for i in range(K):
			for j in range(K):
				et[t,i,j]=alpha[t,i]*beta[t+1,j]*P[i,j]* mvnorm.pdf(X[t+1],mu[int(R[t+1])],sigma[int(R[t+1])])
	
	return et  
	
	
