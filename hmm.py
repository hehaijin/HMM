import numpy as np
from scipy.stats import multivariate_normal as mvnorm
import matplotlib.pyplot as plt


#use the given transition matrix P and mu and sigma to generate a sequence of obervations and their hidden state.
def markovprocess(P,sigma,mu,N):
	p=np.cumsum(P,1)
	zact=np.floor(np.random.rand()*np.shape(mu)[0]).astype(int)
	z=np.array([])
	for i in range(N):
		a=np.random.rand()
		#zact=[min(find(p(zact,:) >a ))];
		zact=np.where(p[zact,:] >a)[0][0] 
		z=np.concatenate((z,np.array([zact]))).astype(int)
	x=np.random.randn(1,N)* sigma[z] + mu[z]
	x=x.reshape((N,))
	return x,z
	
	
	
def dataGenerateion():
	P=np.array([[0.8, 0.1 ,0.1],[0.2, 0.5 , 0.3],[0.3, 0.1, 0.6]])
	mu=np.array([1,2,3])
	N=100
	sigma=np.array([0.3, 0.3, 0.3])
	r=markovprocess(P,sigma,mu,N)
	return r
 	
#normalize a given vector. returns both the nomalized vector and the sum.
def normalize(u):
	N=np.shape(u)[0]
	Result=np.zeros(np.shape(u))
	Z=0
	for i in range(N):
		Z=Z+u[i]
	if Z==0:
		return [u,0]
	for i in range(N):
		Result[i]=u[i]/Z
	return [Result,Z]
	
	
	
u=np.array([1,2,3,4])
r=normalize(u)


#calculating the filtering posterior
#returns alpha and Z()
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

#computes gammer, the smoothing posterior
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
    

#computers the two-sliced marginal
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
	


#the EM algorithm,
def hmmIterate(X,P,pi,mu,sigma):
	T=X.shape[0]
	K=mu.shape[0]
	gamma=forback(P,pi,sigma,mu,X)
	et=eta(P,pi,sigma,mu,X)
	
	ENk1=np.zeros(K)
	ENjk=np.zeors((K,K))
	ENj=np.zeros(K)
	
	for i in range(K):
		ENk1[i]=gamma[0,i]
	
	for j in range(K):
		for k in range(K):
			ENjk[j,k]=0
			for i in range(1,T):
				ENjk[j,k]=ENjk[j,k]+ et[i,j,k]
	
	for j in range(K):
		ENj[j]=0
		for k in range(K):
			ENj[j]=ENj[j]+gamma[j,k]
	
		
	
	
	
	#recalculate transition matrix P
	NN=np.zeros((K,K))
	P=np.zeros((K,K))
	for j in range(K):
		for k in range(K):
			NN[j,k]=sum(et[:,j,k])
	for i in range(K):
		NN[i,:],_=normalize(NN[i,:])
	
	
	#recalculate pi
	npi=np.zeros(K)
	npi=ENk1
		
	
	
	
	#recalculate mu
	nmu=np.zeros(K)
	for i in range(K):
		for j in range(T):
			nmu[i]=nmu[i]+gamma[j,i]*X[j]
	for i in range(K):
		nmu[i]=nmu[i]/ENj[i]
		
			
	
	
	#recalculate sigma
	nsigma=np.zeros(K)
	for i in range(K):
		for j in range(T):
			nsigma[i]=nsigma[i]+gamma[j,i]*X[j]*X[j]
		nsigma[i]=nsigma[i]/ENj[i]-nmu[i]*nmu[i]
	

	
	#calculate loglik
	loglik=0
	for i in range(K):
		loglik=loglik+npi(i)*ENk1[i]
	for j in range(K):
		for k in range(K):
			loglik=loglik+NN[j,k]*ENjk[j,k]
	for i in range(T):
		for j in range(K):
			loglik=loglik+gamma[i,j]*mvnorm.pdf(X[i],nmu[j],nsigma[j])
	
	
	
	return NN,nmu,nsigma,npi,loglik


def hmmRun():
	data,_=dataGenerateion()
	



#implementing the Viterbi algorithm
def Viterbi(P,pi,mus,coMs,X):
	K=P.shape[0]
	N=X.shape[0]
	delta=np.zeros((N,K))
	A=np.zeros((N,K))
	fi=np.zeros((N,K))
	for i in range(N):
		for j in range(K):
			fi[i,j]=mvnorm.pdf(X[i],mean=mus[j],cov=coMs[j])
	delta[0]=np.multiply(pi,fi[0])
	delta[0],_=normalize(delta[0])
	#delta
	for i in range(1,N):
		for j in range(K):
			for k in range(K):
				t=delta[i-1,k]*P[j,k]*fi[i,j]
				if t> delta[i,k]:
					delta[i,k]=t
		delta[i],_=normalize(delta[i])

	#A the matrix that keep track of most likely previous state
	for i in range(1,N):
		for j in range(K):
			m=0
			t=0
			for k in range(K):				
				t1=delta[i-1,j]*P[j,k]*fi[i,j]
				if t1 > t:
					t=t1
					m=k
			A[i,j]=m
			
	Z=np.zeros(N)
	arg=0
	maxi=0
	for j in range(K):
		if delta[N-1,j]> maxi:
			maxi= delta[N-1,j]
			arg=j
	Z[N-1]=arg
	
	
	for i in range(N-2,-1,-1):
		Z[i]=A[i+1,int(Z[i+1])]
	return Z
	
	
def testGamma():
	P=np.array([[0.8, 0.1 ,0.1],[0.2, 0.5 , 0.3],[0.3, 0.1, 0.6]])
	mu=np.array([1,2,3])
	N=100
	pi=np.array([0.2,0.3,0.3])
	sigma=np.array([0.3, 0.3, 0.3])
	X,Z=markovprocess(P,sigma,mu,N) 


	gamma=Viterbi(P,pi,sigma,mu,X)
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
	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

def testAlpha():
	P=np.array([[0.8, 0.1 ,0.1],[0.2, 0.5 , 0.3],[0.3, 0.1, 0.6]])
	mu=np.array([1,2,3])
	N=100
	pi=np.array([0.2,0.3,0.3])
	sigma=np.array([0.3, 0.3, 0.3])
	X,Z=markovprocess(P,sigma,mu,N) 


	R=Viterbi(P,pi,mu,sigma,X)
	#et=eta(P,pi,sigma,mu,X)
	K=P.shape[0]
	T=X.shape[0]

	R=np.zeros(N)
	for i in range(N):
		m=max(alpha[i,:])
		for j in range(K):
			if alpha[i,j]==m:
				R[i]=j

	plt.subplot(211)
	plt.plot(Z)
	plt.subplot(212)
	plt.plot(R)
	plt.show()

def testBeta():
	P=np.array([[0.8, 0.1 ,0.1],[0.2, 0.5 , 0.3],[0.3, 0.1, 0.6]])
	mu=np.array([1,2,3])
	N=100
	pi=np.array([0.2,0.3,0.3])
	sigma=np.array([0.3, 0.3, 0.3])
	X,Z=markovprocess(P,sigma,mu,N) 


	beta=backward(P,pi,sigma,mu,X)
	#et=eta(P,pi,sigma,mu,X)
	K=P.shape[0]
	T=X.shape[0]

	R=np.zeros(N)
	for i in range(N):
		m=max(beta[i,:])
		for j in range(K):
			if beta[i,j]==m:
				R[i]=j

	plt.subplot(211)
	plt.plot(Z)
	plt.subplot(212)
	plt.plot(R)
	plt.show()
	
	
def testViterbi():
	P=np.array([[0.8, 0.1 ,0.1],[0.2, 0.5 , 0.3],[0.3, 0.1, 0.6]])
	mu=np.array([1,2,3])
	N=100
	pi=np.array([0.2,0.3,0.3])
	sigma=np.array([0.3, 0.3, 0.3])
	X,Z=markovprocess(P,sigma,mu,N) 


	R=Viterbi(P,pi,mu,sigma,X)
	#et=eta(P,pi,sigma,mu,X)
	
	plt.subplot(211)
	plt.plot(Z)
	plt.subplot(212)
	plt.plot(R)
	plt.show()
	
testViterbi()
