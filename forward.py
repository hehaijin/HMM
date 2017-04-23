import numpy as np
from normalize import normalize
from scipy.stats import multivariate_normal as mvnorm
from data import markovprocess

def forward(P,pi,sigma,mu,X):
	#P is the transition matrix.
	#L is the local evidence vector L[i,j] means 
	#T deduced from X
	#X as observed data.
	#the resulting alpha array: alpha(n,j) means with obervation 1-n, the hidden zn=j
	K=P.shape[0]
	T=X.shape[0]
	
	#initialize
	alpha=np.zeros((T,K))
	Z=np.zeros(T)
	L=np.zeros((T,K))
	for i in range(T):
		for j in range(K):
			L[i,j]=mvnorm.pdf(X[i],mean=mu[j],cov=sigma[j])
			

	
	[alpha[0,:],Z[0]]=normalize(np.multiply(L[0,:],pi))
	for i in range(1,T):
		[alpha[i,:],Z[i]]=normalize(np.multiply(L[i,:],P.transpose().dot(alpha[i-1,:])))
	
	return (alpha,Z)
	
	


