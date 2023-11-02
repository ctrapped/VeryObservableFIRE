import numpy as np
import sys
#import tables
from scipy.spatial import cKDTree
#from gadget import *
from numba import jit, vectorize, float32, float64


def FindShieldLength(x,rho):
    # build tree for ALL particles..
    tree = cKDTree(x)
    print("Tree Built!")
    des_ngb = 32
    L = np.zeros(len(x))
    
    for ii in xrange(len(x)):
        if not(ii%1e5): print(str(ii)+" of "+str(len(x))+" particles done.")
        ngbdist, ngb = tree.query(x[ii,:], des_ngb)
        h = HsmlIter(ngbdist)
        q = ngbdist/h # how many smoothinglengths away are they?
        K = CubicSpline(q)
        weights = K/np.sum(K) # len=32, cubic spline weight of neighbors
        df = DF(rho[ngb], ngb) # len=32, change in rho from particle to neighbors
        dx = x[ngb] - x[ii,:] # len=32,3 distance along x,y,z from center particle
        ###### Magic
        dx_matrix = np.einsum('j,jk,jl->kl', weights, dx, dx)
        dx_matrix = np.linalg.inv(dx_matrix)
        dweights = np.einsum('kl,jl,j->jk',dx_matrix, dx, weights)
        gr =  np.einsum('jk,j->k',dweights,df)
        
        #dx_matrix = dxMatrix(weights,dx)
        #dx_matrix = np.linalg.inv(dx_matrix) #getMatrixInverse(dx_matrix)
        #dweights = dWeights(dx_matrix, dx, weights)
        #gr = GradRho(dweights,df)
        ##################
        #print gra
        L[ii] =  (rho[ii]/np.sum(gr**2)**0.5 + h/32.**(1./3))/2.
    
    
    return L
@jit
def dWeights(dx_matrix, dx, weights):
    dweights = np.zeros(shape=(weights.shape[0],dx.shape[1]))
    for k in xrange(weights.shape[0]):
        for i in xrange(3):
            dweights[k,i] = weights[k] * np.sum(dx_matrix[i,:]*dx[i,:])
    return dweights
@jit
def GradRho(dweights,df):
    gr = np.zeros(dweights.shape[1])
    for i in xrange(dweights.shape[1]):
        gr[i] = np.sum(dweights[:,i]*df)
    return gr

@jit
def dxMatrix(weights, dx):
    des_ngb=32
    dx_matrix = np.zeros(shape=(dx.shape[1],dx.shape[1]))
    for j in xrange(3):
        for i in xrange(3):
            dx_matrix[i,j]=np.sum(weights*dx[:,i]*dx[:,j])
    return dx_matrix
                        

@vectorize([float32(float32), float64(float64)])
def CubicSpline(q):
    if q <= 0.5:
        return 1 - 6*q**2 + 6*q**3
    elif q <= 1.0:
        return 2 * (1-q)**3
    else: return 0.0

@jit
def HsmlIter(neighbor_dists,  dim=3, error_norm=1e-6):
    if dim==3:
        norm = 32./3
        des_ngb = 32
    elif dim==2:
        norm = 40./7
    else:
        norm = 8./3

    hsml = 0.
    n_ngb = 0.0
    bound_coeff = (1./(1-(2*norm)**(-1./3)))

    upper = neighbor_dists[des_ngb-1] * bound_coeff
    lower = neighbor_dists[1]
    error = 1e100
    count = 0
    while error > error_norm:
        h = (upper + lower)/2
        n_ngb=0.0
        dngb=0.0
        q = 0.0
        for j in range(des_ngb):
            q = neighbor_dists[j]/h
            if q <= 0.5:
                n_ngb += (1 - 6*q**2 + 6*q**3)
            elif q <= 1.0:
                n_ngb += 2*(1-q)**3
        n_ngb *= norm
        if n_ngb > des_ngb:
            upper = h
        else:
            lower = h
        error = np.fabs(n_ngb-des_ngb)
    hsml = h
    return hsml

@jit
def DF(f, ngb):
    df = np.empty(ngb.shape)
    for j in range(ngb.shape[0]):
        df[j] = f[j] - f[0]
    return df
@jit
def transposeMatrix(m):
    return map(list,zip(*m))
@jit
def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]
@jit
def getMatrixDeternminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]
    
    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    return determinant
@jit
def getMatrixInverse(m):
    determinant = getMatrixDeternminant(m)
    #special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]
    
    #find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors

def Reconstruct_ShieldLengths(x,m,rho,h,boxsize):
    # Recontructs the shield lengths in gizmo/FIRE using M. Grudic's MESHOID routine.
    #  Inputs:
    #      x - coords (N,3) x,y,z of particles in snapshot
    #      m - masses (N)
    #      h - smoothing lengths of particles (N)
    
    import meshoid
    
    M = meshoid.FromLoadedSnapshot(x,m,h,boxsize=boxsize) # or use mh?
    
    gradrho = np.sum(M.D(rho)**2,axis=1)**0.5 # magnitude of the density gradient
    L = rho/gradrho + M.h/32.**(1./3) # column density esimate
    
    return L

