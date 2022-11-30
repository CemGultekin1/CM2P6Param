import numpy as np


class Compressor:
    def __init__(self,buffersize) -> None:
        self.buffer_size = buffersize
        self.buffer = []
        self.address = []
        self.weights = []
        self.svd_outputs = None
        self.sides = []
        self.tol = 1e-5

    def land_fill(self,ff,fillbss):
        f = ff.copy()
        iL = np.where(np.isnan(f))[0]
        iW = np.where(~np.isnan(f))[0]
        if np.size(iL) == 0 :
            return f
        pWW = fillbss[:,iW].T@fillbss[:,iW]
        pWL = fillbss[:,iW].T@fillbss[:,iL]
        pLL = fillbss[:,iL].T@fillbss[:,iL]

        fW = f[iW]
        mat = np.concatenate([pWL,np.eye(len(iL)) - pLL ],axis= 0)
        rhs1 = fW - pWW@fW
        rhs2 = pWL.T@fW
        rhs = np.concatenate([rhs1,rhs2],axis= 0)
        matq,matr = np.linalg.qr(mat,mode = 'reduced')
        f[iL] = np.linalg.inv(matr)@matq.T@rhs
        return f

    def tol_cut_svd(self,fs):
        u,s,vh = np.linalg.svd(fs,full_matrices = False)
        rs = s/s[0]
        I = rs > self.tol
        return u[:,I],s[I],vh[I,:]
    def get_weights(self,f):
        vh = self.svd_outputs['vh']
        return vh@f
    def get_filter(self,f):
        vh = self.svd_outputs['vh']
        return vh.T@f
    def process(self,f,ad):
        f = f.reshape([-1])
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(f)
            self.address.append(ad)
            return
        if self.svd_outputs is None:
            wet_filters = [f for f in self.buffer if not np.any(np.isnan(f))]
            _,_,uhwet = self.tol_cut_svd(np.stack(wet_filters,axis = 0))
            sb = wet_filters+[self.land_fill(f,uhwet) for f in self.buffer if np.any(np.isnan(f))]
            print('nwet :',str(len(wet_filters)),', nmix :',str(len(sb) - len(wet_filters)))
            sb = np.stack(sb,axis = 0)
            u,s,vh = self.tol_cut_svd(sb)
            self.svd_outputs = dict(
                u = u,s = s, vh = vh
            )
            self.weights = [vh@s for s in sb]
        wf = self.land_fill(f,self.svd_outputs['vh'])
        w_wf = self.get_weights(wf)
        self.weights.append(w_wf)
        self.address.append(ad)
    def reshape2square(self,f):
        x = np.sqrt(f.shape[0]).astype(int)
        return f.reshape([x,x])
    def rebuild_filter(self,i):
        if i >= len(self.weights):
            return None
        return self.reshape2square(self.get_filter(self.weights[i]))
    def gcm_weights(self,lats,lons):
        locs = []
        for z,(i,j) in enumerate(self.address):
            if i in lats and j in lons:
                locs.append(z)
        assert len(locs) == len(lats)*len(lons)
        ws = np.stack([self.weights[z] for z in locs],axis= 0)
        return ws

        
        


        

            


        