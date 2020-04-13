# _*_ coding: utf-8 _*_

# Copyright (c) 2020 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
Deal vectorial fields, like calculating stream function, velocity potential.

Primary functions of the Li et al. (2006) method
Li et al. (2006) minimization method

refer to https://raw.githubusercontent.com/tiagobilo/vector_fields/2afc4311194340e0427829e2eb617d4036d29d2d/psi_phi.py
"""

import numpy as np
import scipy.optimize as optimize
import time


# Function to be minimized: Objective functional + Tikhonov's regularization term 
def ja(x,y,DX,DY,M1,N1,IDATA,ZBC,MBC,ALPHA):

	"""
	Fitting function from Li et al (2006) method
	"""

	# Derive velocity from PSI and PHI
	Ax = derive_ax(x,DX,DY,M1,N1,IDATA) 

	# "Error" to be minimized
	e = y.copy()-Ax

	# Matrices multiplications
	Mat1 = np.matmul(e.T,e)
	Mat2 = np.matmul(x.T,x)

	# Tikhionov's functional
	J = np.dot(0.5,Mat1) + np.dot(ALPHA*0.5,Mat2)

	return J


# Gradient of ja (i. e., Jacobian of ja)
# following Li et al method A.T(y-Ax) + alpha x.
# In our case, since Ax compute the velocity
# from the psi and phi, A.T(y-Ax) will be
# the curl and -divergent of the velocity difference 
# i.e., (zeta_o - zeta_r) and (div_r - div_o) 
def grad_ja(x,y,DX,DY,M1,N1,IDATA,ZBC,MBC,ALPHA):

	"""
	Jacobian of the fitting function ja from Li et al (2006) method
	"""

	# Derive velocity from PSI and Qi
	Ax = derive_ax(x,DX,DY,M1,N1,IDATA) 

	# "Error" to be minimized
	e = y-Ax

	# Compute adjoint term 
	# i. e., velocity difference curl and
	# velocity difference divergence
	adj = derive_adj(e,DX,DY,M1,N1,ZBC,MBC,IDATA)

	# Jacobian
	gj = -adj+np.dot(ALPHA,x)

	return gj



# Derive velocity components 
# from psi phi field 
def derive_ax(x,DX,DY,M1,N1,IDATA):

	"""
	Derive velocity from psi and phi fields for the Li et al (2006) method
	"""

	## Re-organize x
	psi = x[:M1*N1].reshape((M1,N1))
	phi = x[M1*N1:].reshape((M1,N1))

	## Derivation
	dpsidy = (psi[1:,:]-psi[:-1,:])/((DY[1:,:]+DY[:-1,:])/2.0)
	dpsidx = (psi[:,1:]-psi[:,:-1])/((DX[:,1:]+DX[:,:-1])/2.0)

	dphidy = (phi[1:,:]-phi[:-1,:])/((DY[1:,:]+DY[:-1,:])/2.0)
	dphidx = (phi[:,1:]-phi[:,:-1])/((DX[:,1:]+DX[:,:-1])/2.0)

	u = ((dpsidy[:,1:]+dpsidy[:,:-1])/2.0) + ((dphidx[1:,:] + dphidx[:-1,:]) /2.0)
	v = (-(dpsidx[1:,:]+dpsidx[:-1,:])/2.0) + ((dphidy[:,1:] + dphidy[:,:-1]) /2.0)

	# Organize the variables
	ax = np.ones(2*(M1-1)*(N1-1))*np.nan
	ax[:int(ax.shape[0]/2)] = u.reshape((M1-1)*(N1-1))
	ax[int(ax.shape[0]/2):] = v.reshape((M1-1)*(N1-1))

	# Remove NaNs
	ax = ax[IDATA]

	return ax


# Derive the adjoint term
# (i. e., relative vorticity)
def derive_adj(e,DX,DY,M1,N1,ZBC,MBC,IDATA):

	"""
	Derive the adjoint term of the Li et al (2006) method
	"""

	## Resized error
	er = np.zeros(2*(M1-1)*(N1-1))
	er[IDATA] = e.copy()

	## Re-organize variables
	# Velocity
	u = er[:int(er.shape[0]/2)]
	v = er[int(er.shape[0]/2):]	
	
	u = u.reshape((M1-1,N1-1))
	v = v.reshape((M1-1,N1-1))

	# Spatial resolution
	dy = (DY[1:-1,1:]+DY[1:-1,:-1])/2.0
	dx = (DX[1:,1:-1]+DX[:-1,1:-1])/2.0

	## Derivation of the curl and divergence
	# Curl terms
	dudy = (u[1:,:]-u[:-1,:])/dy
	dudy = (dudy[:,1:]+dudy[:,:-1])/2.0

	dvdx = (v[:,1:]-v[:,:-1])/dx
	dvdx = (dvdx[1:,:]+dvdx[:-1,:])/2.0

	# Divergent terms
	dvdy = (v[1:,:]-v[:-1,:])/dy
	dvdy = (dvdy[:,1:]+dvdy[:,:-1])/2.0

	dudx = (u[:,1:]-u[:,:-1])/dx
	dudx = (dudx[1:,:]+dudx[:-1,:])/2.0	


	# Curl
	curl = dvdx-dudy
	curl1 = np.ones((M1,N1))*np.nan
	curl1[1:-1,1:-1] = curl


	# Divergence
	div = dudx+dvdy
	div1 = np.ones((M1,N1))*np.nan
	div1[1:-1,1:-1] = div	


	## Calculate boundary conditions for 
	## the curl and divergence fields
	if ZBC == 'periodic' or MBC == 'periodic':

		if ZBC == 'periodic':

			# Curl
			dudy_1 = (u[1:,0]-u[:-1,0])/dy[:,0]
			dudy_2 = (u[1:,-1]-u[:-1,-1])/dy[:,-1]

			dvdx_1 = (v[:,0]-v[:,-1])/dx[:,0]
			dvdx_2 = dvdx_1.copy() 

			curl1[1:-1,0] = ((dvdx_1[1:]+dvdx_1[:-1])/2.0)-dudy_1
			curl1[1:-1,-1] = ((dvdx_2[1:]+dvdx_2[:-1])/2.0)-dudy_2


			# Divergent
			dvdy_1 = (v[1:,0]-v[:-1,0])/dy[:,0]
			dvdy_2 = (v[1:,-1]-v[:-1,-1])/dy[:,-1]

			dudx_1 = (u[:,0]-u[:,-1])/dx[:,0]
			dudx_2 = dudx_1.copy()

			div1[1:-1,0] = ((dudx_1[1:]+dudx_1[:-1])/2.0)+dvdy_1
			div1[1:-1,-1] = ((dudx_2[1:]+dudx_2[:-1])/2.0)+dvdy_2

		else:
			curl1[:,0] = curl1[:,1]; curl1[:,-1] = curl1[:,-2]
			div1[:,0] = div1[:,1]; div1[:,-1] = div1[:,-2]


		if MBC == 'periodic':

			# Curl
			dudy_1 = (u[0,:]-u[-1,:])/dy[0,:]
			dudy_2 = dudy_1.copy()

			dvdx_1 = (v[0,1:]-v[0,:-1])/dx[0,:]
			dvdx_2 = (v[-1,1:]-v[-1,:-1])/dx[-1,:]

			curl1[0,1:-1] = dvdx_1-((dudy_1[1:]+dudy_1[:-1])/2.0)
			curl1[-1,1:-1] = dvdx_2-((dudy_2[1:]+dudy_2[:-1])/2.0)


			# Divergent 
			dvdy_1 = (v[0,:]-v[-1,:])/dy[0,:]
			dvdy_2 = dvdy_1.copy()

			dudx_1 = (u[0,1:]-u[0,:-1])/dx[0,:]
			dudx_2 = (u[-1,1:]-u[-1,:-1])/dx[-1,:]

			div1[0,1:-1] = dudx_1 + ((dvdy_1[1:]+dvdy_1[:-1])/2.0)
			div1[-1,1:-1] = dudx_2 + ((dvdy_2[1:]+dvdy_2[:-1])/2.0)			

		else:

			curl1[0,:] = curl1[1,:]; curl1[-1,:] = curl1[-2,:]
			div1[0,:] = div1[1,:]; div1[-1,:] = div1[-2,:]

	else:

		# All closed edges (i.e., land edges)
		curl1[0,1:-1] = curl[0,:]; curl1[-1,1:-1] = curl[-1,:]
		curl1[:,0] = curl1[:,1]; curl1[:,-1] = curl1[:,-2]

		div1[0,1:-1] = div[0,:]; div1[-1,1:-1] = div[-1,:]
		div1[:,0] = div1[:,1]; div1[:,-1] = div1[:,-2]



	# Organize the variables
	curl = curl1.reshape(M1*N1)
	div = div1.reshape(M1*N1)

	adj = np.ones(2*M1*N1)*np.nan
	adj[:M1*N1] = curl
	adj[M1*N1:] = -div

	return adj


### Auxiliary Functions
## Cumulative velocity integration
def v_zonal_integration(V,DX):
	"""
	Zonal cumulative integration of velocity
	in a rectangular grid using a trapezoidal
	numerical scheme. 

	- Integration occurs from east to west

	- Velocity is assumed to be zero at the lateral
	boundaries defined by NaN.

	Input:
			V   [M,N]: meridional velocity component in m s-1
			DX 	[M,N-1]: zonal distance in m

	Output: 
			vi [M,N]: Integrated velocity in m2 s-1

	"""

	## Zero velocity at the boundaries
	v = V.copy()

	ibad = np.isnan(v)
	v[ibad] = 0.0

	## Zonal integration	
	vi = np.zeros(v.shape)

	for j in range(2,vi.shape[1]+1):
		vi[:,-j] = np.trapz(v[:,-j:], dx=DX[:,(-j+1):])

	return vi


def v_meridional_integration(V,DY):
	"""
	Meridional cumulative integration of velocity
	in a rectangular grid using a trapezoidal
	numerical scheme. 

	- Integration occurs from north to south

	- Velocity is assumed to be zero at the lateral
	boundaries defined by NaN.

	Input:
			V   [M,N]: meridional velocity component in m s-1
			DY 	[M-1,N]: zonal distance in m

	Output: 
			vi [M,N]: Integrated velocity in m2 s-1

	"""

	## Zero velocity at the boundaries
	v = V.copy()

	ibad = np.isnan(v)
	v[ibad] = 0.0

	## Zonal integration	
	vi = np.zeros(v.shape)

	for i in range(2,vi.shape[0]+1):
		vi[-i,:] = np.trapz(v[-i:,:],dx=DY[(-i+1):,:],axis=0)

	return vi


## Calculate distances from a rectangular lat/lon grid
def dx_from_dlon(lon,lat):
	"""
	Calculate zonal distance at the Earth's surface in m from a 
	longitude and latitude rectangular grid

	Input:
			lon [M,N]: Longitude in degrees
			lat [M,N]: Latitude in degrees

	Output:
			dx   [M,N-1]: Distance in meters

	"""

	# Earth Radius in [m]
	earth_radius = 6371.0e3


	# Convert angles to radians
	lat = np.radians(lat)
	lon = np.radians(lon)
	
	# Zonal distance in radians
	dlon = np.diff(lon,axis=1)
	lat = (lat[:,1:]+lat[:,:-1])/2.0


	# Zonal distance arc 
	a = np.cos(lat)*np.cos(lat)*(np.sin(dlon/2.0))**2.0
	angles = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

	# Distance in meters
	dx = earth_radius * angles

	return dx


def dy_from_dlat(lat):
	"""
	Calculate meridional distance at the Earth's surface in m from a 
	longitude and latitude rectangular grid

	Input:
			lat [M,N]: Latitude in degrees

	Output:
			dy   [M-1,N]: Distance in meters

	"""

	## Meridional resolution (m)
	dy = np.diff(lat,axis=0)
	dy = dy*111194.928

	return dy


def psi_lietal(IPSI,IPHI,DX,DY,U,V,ZBC='closed',MBC='closed',ALPHA=1.0e-14):

	"""
	Compute streamfunction implementing Li et al. (2006) method. Its advantages consist in
	extract the non-divergent and non-rotational part of the flow without explicitly applying boundary 
	conditions and computational efficiency.

	This method also minimizes the difference between the reconstructed and original velocity fields. Therefore 
	it is ideal for large and non-regular domains with complex coastlines and islands.

	Streamfunction and velocity potential are staggered with the velocity components.

	Input:
			IPSI [M,N]		: Streamfunction initial guess
			IPHI [M,N]      : Velocity potential initial guess
			DX 	 [M,N]      : Zonal distance (i.e., resolution) 
			DY   [M,N]      : Meridional distance (i.e., resolution)
			U    [M-1,N-1]  : Original zonal velocity field defined between PSI and PHI grid points 
			V    [M-1,N-1]  : Original meridional velocity field defined between PSI and PHI grid points

	Optional Input:
			ZBC				: Zonal Boundary Condition for domain edge (closed or periodic)
			MBC				: Meridional Boundary Condition for domain edge (closed or periodic)
			ALPHA 			: Regularization parameter

	Output:
			psi [M,N]		: Streamfunction
			phi [M,N]		: Velocity Potential


	Obs1: PSI and PHI over land and boundaries have to be 0.0
	for better performance. However U and V can be masked with 
	NaNs 

	Obs2: Definitions

	U = dPsi/dy + dPhi/dx
	V = -dPsi/dx + dPhi/dy 

	Obs3: BCs are applied only to the Jacobian of the
	minimization function and are referred to the edges
	of the rectangular domain. 

	:Examples:
		M = 64
		N = 64
		IPSI = np.zeros((M, N))
		IPHI = np.zeros((M, N))
		DX = np.zeros((M, N)) + 2.5
		DY = np.zeros((M, N)) + 2.5
		U = np.random.rand(M-1, N-1)
		V = np.random.rand(M-1, N-1)
		psi,phi = psi_lietal(IPSI,IPHI,DX,DY,U,V)
	"""

	## Reshape/Resize variables ("Vectorization")
	# Velocity
	M,N = U.shape

	y = np.ones(2*M*N)*np.nan

	# Velocity y = (U11,U12,...,U31,U32,....,V11,V12,....)
	y[:int(y.shape[0]/2)] = U.reshape(M*N)
	y[int(y.shape[0]/2):] = V.reshape(M*N)

	idata = ~np.isnan(y.copy())
	y = y[idata]


	# Stream function and velocity potential
	M1,N1 = IPSI.shape

	x = np.ones(2*M1*N1)*np.nan

	# PSI and PHI vector: x = (PSI11,PSI12,...,PSI31,PSI32,...,PHI11,PHI12,...)
	x[:int(x.shape[0]/2)] = IPSI.reshape(M1*N1)
	x[int(x.shape[0]/2):] = IPHI.reshape(M1*N1)		


	print('       Optimization process')
	t0 = time.clock()
	pq = optimize.minimize(ja,x,method='L-BFGS-B',jac=grad_ja,
		args=(y,DX,DY,M1,N1,idata,ZBC,MBC,ALPHA),options={'gtol': 1e-16})

	t1 = time.clock()

	print('           Time for convergence: %1.2f min'%((t1-t0)/60.0))
	print('           F(x): %1.2f'%(pq.fun))		

	psi = pq.x[:int(x.shape[0]/2)].reshape((M1,N1))
	phi = pq.x[int(x.shape[0]/2):].reshape((M1,N1))

	return psi,phi

