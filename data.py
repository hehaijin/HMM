import numpy as np

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
	
	
	
	
	

P=np.array([[0.8, 0.1 ,0.1],[0.2, 0.5 , 0.3],[0.3, 0.1, 0.6]])
mu=np.array([1,2,3])
N=100
sigma=np.array([0.3, 0.3, 0.3])
r=markovprocess(P,sigma,mu,N) 
print(r)


