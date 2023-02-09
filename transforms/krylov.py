import math
import numpy as np
class iterative_inversion:
    def __init__(self,degree) -> None:
        self.degree = degree
        pass
    def invert(self,forward_pass:callable,pseudo_inverse:callable,x0):
        '''
        Returns approximate inverse of x0
        '''
        x = x0.copy()
        m = self.degree
        y = math.comb(m, 1 + 0)*math.pow(-1,0)*x
        for i in range(1,m):
            xinv = pseudo_inverse(x)
            x = forward_pass(xinv)
            y += math.comb(m, 1 + i)*math.pow(-1,i)*x
        xinv = pseudo_inverse(y)
        return xinv
    
class growing_orthogonals_decomposition:
    def __init__(self) -> None:
        self.qmat = None
        self.rmat = None
    def q_orthogonal(self,v:np.ndarray):
        if self.qmat is None:
            return None,v
        r = self.qmat.T@v
        return r,v - self.qmat@r
    def r_grow(self,r,n):
        if r is None:
            return np.eye(1)*n
        m = self.rmat.shape[0] + 1
        rr = np.empty((m,m))
        rr[:-1,:-1] = self.rmat
        rr[-1,:] = 0
        rr[:-1,-1] = r
        rr[-1,-1] = n
        return rr
    def q_grow(self,v):
        v = v.reshape([-1,1])
        if self.qmat is None:
            return v
        return np.concatenate([self.qmat,v],axis = 1)
    def __len__(self,):
        if self.qmat is None:
            return 0
        return self.qmat.shape[1]
    def add(self,v):    
        n0 = np.linalg.norm(v)
        r,v_orth =self.q_orthogonal(v)
        n = np.linalg.norm(v_orth)

        relnorm = n/n0
        if relnorm < 1e-5:
            return False
        v_orth = v_orth/n
        self.qmat = self.q_grow(v_orth)
        self.rmat = self.r_grow(r,n)
        return True
    def res(self,v):
        return self.q_orthogonal(v)[1]
    def nres(self,v):
        return np.linalg.norm(self.res(v))
    def solve(self,v):
        return np.linalg.solve(self.rmat,self.qmat.T@v)
    def orthogonality(self,):
        return np.linalg.norm(self.qmat.T @ self.qmat - np.eye(self.qmat.shape[1]))

class krylov_inversion(growing_orthogonals_decomposition):
    def __init__(self,maxiter,reltol,implicit_matmultip) -> None:
        super().__init__()
        self.maxiter = maxiter
        self.reltol = reltol
        self.implicit_matmultip = implicit_matmultip
        self.iterates = []
    def add(self, x):
        y = self.implicit_matmultip(x)
        if super().add(y):
            self.iterates.append(x)
    def solve(self,y:np.ndarray):
        self.add(y)
        nres = [self.nres(y)]
        i = 0
        while i < self.maxiter and self.reltol < nres[-1]/nres[0]:
            x = self.implicit_matmultip(self.qmat[:,-1])
            self.add(x)
            nres.append(self.nres(y))
            i+=1
            if nres[-1]/nres[0] > 1:
                break
        coeffs = super().solve(y)
        return np.stack(self.iterates,axis=1)@coeffs



class two_parts_krylov_inversion(growing_orthogonals_decomposition):
    def __init__(self,maxiter,reltol,matmultip_1,matmultip_2,transform_1_2) -> None:
        super().__init__()
        self.maxiter = maxiter
        self.reltol = reltol
        self.matmultip_1 = matmultip_1
        self.matmultip_2 = matmultip_2
        self.transform_1_2 = transform_1_2
        self.sources = []
        self.iterates = []
        
    def add(self, x1,):
        y1 = self.matmultip_1(x1)
        x2 = self.transform_1_2(x1)
        y2 = self.matmultip_2(x2)

        if super().add(y1):
            self.iterates.append(x1)
            self.sources.append(1)
        if super().add(y2):
            self.iterates.append(x2)
            self.sources.append(2)
            
    def solve(self,y:np.ndarray,):
        self.add(y)
        nres = [self.nres(y)]
        i = 0
        while i < self.maxiter and self.reltol < nres[-1]/nres[0]:
            self.add(self.qmat[:,len(self.sources) -1 - self.sources[::-1].index(1)])
            nres.append(self.nres(y))
            i+=1
            if nres[-1]/nres[0] > 1:
                break
        coeffs = super().solve(y)
        src = np.array(self.sources)
        x1 = np.stack([self.iterates[i] for i,j in enumerate(src) if j == 1],axis = 1)
        x2 = np.stack([self.iterates[i] for i,j in enumerate(src) if j == 2],axis = 1)
        x1star = x1@coeffs[src == 1]
        x2star = x2@coeffs[src == 2]
        return x1star,x2star


        
def test_krylov_inversion():
    np.random.seed(0)
    d = 256
    mat = np.random.randn(d,d)
    q,r = np.linalg.qr(mat)
    signdiag = np.diag( (np.diag(r) > 0)*2 -1 ) 
    q = q@signdiag
    r = signdiag@r
    qrerr = np.sum(np.square(q@r - mat))
    print(f'qrerr = {qrerr}')

    god = growing_orthogonals_decomposition()

    
    for i in range(mat.shape[1]):
        god.add(mat[:,i])
    qerr = np.sum(np.square(god.qmat - q))
    rerr = np.sum(np.square(god.rmat - r))
    print(f'qerr,rerr:{qerr,rerr}')
    print(f'orthogonality:\t{god.orthogonality()}')

    y = np.random.randn(d)
    god = growing_orthogonals_decomposition()

    
    def matmultip(x_):
        return mat@x_
    gres = krylov_inversion(d,1e-2,matmultip)
    sltn = gres.solve(y)

def main():
    np.random.seed(0)
    d1 = 256
    r = 128
    d2 = 128
    mat1 = np.random.randn(d1,r)@np.random.randn(r,d1)
    mat2 = np.random.randn(d1,d2)
    tra12 = np.random.randn(d2,d1)
    y = np.random.randn(d1)
    def matmultip1(x_):
        return mat1@x_
    def matmultip2(x_):
        return mat2@x_
    def trans12(x):
        return tra12@x
    tpk = two_parts_krylov_inversion(d1,1e-2,matmultip1,matmultip2,trans12)
    x1,x2 = tpk.solve(y)
    err = np.linalg.norm(matmultip1(x1) + matmultip2(x2) - y)
    print(f'err:\t{err}')
    
    
if __name__ == '__main__':
    main()