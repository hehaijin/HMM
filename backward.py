
import numpy as np
from scipy.stats import multivariate_normal as mvnorm

def backward(P,pi,sigma,mu,X):
	
	T=X.shape[0]
	K=mu.shape[0]
	
	#initialize beta 
	beta=np.zeros((T,K))
	for i in range(K):
		beta[T-1,i]=1
	
	#initialize L 
	L=np.zeros((T,K))
	for i in range(T):
		for j in range(K):
			L[i,j]=mvnorm.pdf(X[i],mean=mu[j],cov=sigma[j])	
	
	
	for i in range(T-2,-1,-1):
		
		beta[i,:]=P.dot(np.multiply(L[i,:],beta[i+1,:]))
	return beta

