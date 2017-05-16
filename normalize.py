import numpy as np

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

