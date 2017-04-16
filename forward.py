import numpy as np
from normalize import normalize
from scipy.stats import multivariate_normal as mvnorm
from data import markovprocess

def forward(P,pi,sigma,mu,X):
	#P is the transition matrix.
	#L is the local evidence vector L[i,j] means 
	#T 
	#X as observed data.
	K=np.shape(P)[0]
	
	T=np.shape(X)[0]
	
	#initialize
	alpha=np.zeros((T,K))
	Z=np.zeros(T)
	L=np.zeros((T,K))
	for i in range(T):
		for j in range(K):
			L[i,j]=mvnorm.pdf(X[i],mean=mu[j],cov=sigma[j])
			

	
	[alpha[0,:],Z[0]]=normalize(np.multiply(L[0,:],pi))
	for i in range(1,N):
		[alpha[i,:],Z[i]]=normalize(np.multiply(L[i,:],P.dot(alpha[i-1,:])))
	
	return [alpha,Z]
	
	
P=np.array([[0.8, 0.1 ,0.1],[0.2, 0.5 , 0.3],[0.3, 0.1, 0.6]])
mu=np.array([1,2,3])
N=100
pi=np.array([0.2,0.3,0.3])
sigma=np.array([0.3, 0.3, 0.3])
X=markovprocess(P,sigma,mu,N) 
print(X[10])
forward(P,pi,sigma,mu,X)


