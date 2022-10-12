import itertools
from os import remove
import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
# from data.geography import geographic_features2
import xarray as xr
# class GeographicBoundary:
#     def __init__(self,boundaries):


class ClimateData:
    'Characterizes a dataset for PyTorch'
    def __init__(self,data_address,):
        self.ds_data = xr.open_zarr(data_address)
    def read(self,boundaries,fields):
        self.ds_data.sel(time=self.ds_data.time[0].data,\
                xu_ocean=slice(boundaries[0], self.domains['xmax'][nd]),\
                yu_ocean=slice(self.domains['ymin'][nd], self.domains['ymax'][nd]))
        y=self.ds_data.xu_ocean.values
        x=self.ds_data.xu_ocean.values
        lat1, lat2, lng1, lng2 = None,None,None,None#geographic_features2(len(y),rec_field)
        geo = []
        for nd in range(self.num_domains):
            datsel=None

            xx=datsel.xu_ocean.values
            yy=datsel.yu_ocean.values

            i0=np.argmin(np.abs(yy[0]-y))
            i1=np.argmin(np.abs(yy[-1]-y))+1

            j0=np.argmin(np.abs(xx[0]-x))
            j1=np.argmin(np.abs(xx[-1]-x))+1

            locgeo = lat1[i0:i1], lat2[i0:i1], lng1[j0:j1], lng2[j0:j1]
            locgeo = [torch.tensor(ll,dtype=torch.float32) for ll in locgeo]
            geo.append(locgeo)
        return geo
    def receive_scalars(self,dataset:'Dataset'):
        self.inscalars = dataset.inscalars
        self.outscalars = dataset.outscalars
        self.rec_field = dataset.rec_field
        self.pooler = dataset.pooler
        self.geo  = dataset.geo

    def compute_normalization_constants(self,):

        stats = {}
        for key in self.input_fields + self.output_fields:
            mom1 = 0
            mom2 = 0
            M=50
            for t in range(M):
                subds_data  = self.ds_data.isel(time = t)
                vals = subds_data[key].values
                vals[vals!=vals]= 0
                vals = vals[vals!=0]

                if self.normalization == "standard":
                    mom1 += np.mean(vals)
                    mom2 += np.mean(np.square(vals))
                elif self.normalization == "absolute":
                    mom1 += np.mean(vals)*0
                    mom2 += np.mean(np.abs(vals))

            mom1 = mom1/M
            mom2 = mom2/M
            if self.normalization == "standard":
                stats[key] = (mom1,np.sqrt(mom2 - np.square(mom1)))
            elif self.normalization == "absolute":
                stats[key] = (mom1,mom2)
        self.save_affine_maps(stats,)
        return stats
    def set_same_padding(self,flag):
        self.same_padding = flag
    def get_affine_maps(cls,st,inflds):
        avgscalars = [st[key][0] for key in inflds]
        stdscalars = [st[key][1] for key in inflds]
        return avgscalars,stdscalars

    def save_affine_maps(self,stats,):
        self.inscalars = self.get_affine_maps(stats,self.input_fields)
        self.outscalars = self.get_affine_maps(stats,self.output_fields)

    def __len__(self):
        'Denotes the total number of samples'
        return self.num_domains*self.num_time
    def compute_mask(self,x):
        prt = torch.from_numpy(np.stack([x],axis=0))
        prt[prt==prt] = 0.
        prt[prt!=prt] = 1.
        mask = 1.-self.pooler(prt).numpy()[0]
        return mask
    def land_density_map(self,dlat,dlon):

        X,_,_ = self[0]
        X = X[:1]
        X[X==0] = np.nan
        # X = X[:,::-1]
        mask = 1-self.compute_mask(X)
        conv  = nn.Conv2d(1,1,(dlat,dlon))
        conv.weight.data = conv.weight.data*0+1
        conv.bias.data = conv.bias.data*0
        mask = torch.from_numpy(np.stack([mask],axis=0)).type(torch.float32)
        density = conv(mask)/dlat/dlon
        # print('X.shape: ',X.shape)
        # print('mask.shape: ',mask.shape)
        # print('density.shape: ',density.shape)
        return density.detach().numpy()[0,0]
    def most_ocean_points(self,dy,dx,sep,num):
        density = self.land_density_map(dy,dx)
        # import matplotlib.pyplot as plt
        # plt.imshow(density[::-1])
        # plt.savefig('density.png')
        # plt.close()
        # if not np.any(density==0):
        #     return None
        inds = np.dstack(np.unravel_index(np.argsort(density.ravel()), density.shape))[0]
        # inds = inds[::-1,:]
        inds = inds + self.rec_field//2
        lat,lon = self.coords[0]
        spread = self.rec_field//2
        lon=np.concatenate([lon[-spread:],lon,lon[:spread]], axis=0)
        # print(lat.shape,lon.shape,density.shape)
        # print(lat[::10])
        ymin,xmin = lat[inds[:,0]],lon[inds[:,1]]
        ymax,xmax = lat[inds[:,0]+dy],lon[inds[:,1]+dx]
        # print(lat.shape,lon.shape)
        def remove_close_ones(vecs,s,i,d):
            vec = vecs[s]
            I = np.where(np.abs(vec[i]-vec[i+1:])>d)[0]
            if len(I)==0:
                return vecs,False
            I = I +i+1
            for j in range(len(vecs)):
                vec = vecs[j]
                vec = np.concatenate([vec[:i],vec[I]],axis=0)
                vecs[j] = vec
            return vecs,True
        exts = [xmin,xmax,ymin,ymax,inds]
        coords = []
        for i in range(num):
            xmin,xmax,ymin,ymax,inds = exts
            if len(xmin) <= i :
                break
            ind = tuple(inds[i].tolist())
            # print(ind)
            d = density[ind]
            # print(d,xmin[i],xmax[i],ymin[i],ymax[i])
            coords.append([inds[0],inds[1]])
            exts,flag = remove_close_ones(exts,0,i,sep)
            if not flag:
                exts,flag = remove_close_ones(exts,2,i,sep)
        return coords

    def set_receptive_field(self,rec_field):
        self.rec_field = int(rec_field)
        if self.glbl_data:
            self.dimens[1]+=(rec_field//2)*2
        self.pooler=nn.MaxPool2d(self.rec_field,stride=1)
        # self.set_same_padding(True)
        self.geo = self.load_geo(rec_field)
    def domain_index(self,index):
        return index%self.num_domains

    def normalize(self,vec,input=True):
        if input:
            scalars = self.inscalars
        else:
            scalars = self.outscalars
        for i in range(len(vec)):
            u = vec[i]
            vec[i] = (u - scalars[0][i])/scalars[1][i]
        return vec
    def denormalize(self,vec, input= True):
        if input:
            scalars = self.inscalars
        else:
            scalars = self.outscalars
        for i in range(len(vec)):
            u = vec[i]
            vec[i] = u*scalars[1][i] + scalars[0][i]
        return vec
    def to_physical(self,vec,input=True):
        if input:
            if not self.latfeat:
                return self.denormalize(vec,input=input)
            else:
                return self.denormalize(vec[:-2],input=input)
        else:
            return self.denormalize(vec,input=input)

    def get_pure_data(self,domid,timeid):
        index = timeid*self.num_domains + domid
        X,Y,mask = self[index]
        lon,lat = self.coords[domid]
        return X,Y,mask,lon,lat
    def confine(self,nd,xmin,xmax,ymin,ymax,):#index = False):
        # if index:
        #     datsel=self.ds_data.isel(time=self.ds_data.time[0].data, \
        #         xu_ocean=slice(xmin,xmax),\
        #         yu_ocean=slice(ymin,ymax))

        self.domains['xmin'][nd], self.domains['xmax'][nd],\
            self.domains['ymin'][nd], self.domains['ymax'][nd]\
                = xmin,xmax,ymin,ymax
        # lat,lon = self.coords[0]
        # print()
        dimens=[]
        for nd in range(self.num_domains):
            datsel=self.ds_data.sel(time=self.ds_data.time[0].data, \
                xu_ocean=slice(self.domains['xmin'][nd], self.domains['xmax'][nd]),\
                yu_ocean=slice(self.domains['ymin'][nd], self.domains['ymax'][nd]))
            dimens.append([datsel.yu_ocean.shape[0],datsel.xu_ocean.shape[0]])
        dimens=torch.tensor(dimens)
        self.dimens=torch.amax(dimens,dim=0)
        self.glbl_data=self.ds_data.yu_ocean.shape[0]*self.ds_data.xu_ocean.shape[0]==self.dimens[0]*self.dimens[1]
        self.periodic_lon_expand = self.glbl_data

        coords=[]
        icoords=[]

        y=self.ds_data.yu_ocean.values
        x=self.ds_data.xu_ocean.values
        for nd in range(self.num_domains):
            datsel=self.ds_data.sel(time=self.ds_data.time[0].data,\
                xu_ocean=slice(self.domains['xmin'][nd], self.domains['xmax'][nd]),\
                yu_ocean=slice(self.domains['ymin'][nd], self.domains['ymax'][nd]))

            xx=datsel.xu_ocean.values
            yy=datsel.yu_ocean.values

            i0=np.argmin(np.abs(yy[0]-y))
            i1=np.argmin(np.abs(yy[-1]-y))+1

            j0=np.argmin(np.abs(xx[0]-x))
            j1=np.argmin(np.abs(xx[-1]-x))+1

            coords.append([yy,xx])
            icoords.append([i0,i1,j0,j1])

        self.coords=coords
        self.icoords=icoords
        # print(icoords)


    def __getitem__(self,index):
        nd=index%self.num_domains
        nt=np.int64(np.floor(index/self.num_domains))+self.time_st
        datsel=self.ds_data.isel(time=nt).sel(xu_ocean=slice(self.domains['xmin'][nd], self.domains['xmax'][nd]), \
                                    yu_ocean=slice(self.domains['ymin'][nd], self.domains['ymax'][nd]))
        X=[datsel[key].values for key in self.input_fields]
        spread = self.rec_field//2
        # print(f'spread = {spread}')
        if spread > 0:
            # print(spread,datsel[self.output_fields[0]].shape)
            Y=[datsel[key].values[spread:-spread,:] for key in self.output_fields]
            # Y=[y[::-1] for y in Y]

            nouts = len(self.output_fields)
            if not self.periodic_lon_expand:
                Y=[Y[i][:,spread:-spread] for i in range(nouts)]
        else:
            Y=[datsel[key].values for key in self.output_fields]
            # Y=[y[::-1] for y in Y]

        if self.normalization is not None and self.inscalars is not None and self.outscalars is not None:
            X = self.normalize(X,input=True)
            Y = self.normalize(Y,input=False)
        X = np.stack(X,axis = 0)
        Y = np.stack(Y,axis = 0)

        if self.latfeat:
            locgeo=self.geo[nd]
            xx=np.ones(X[:1].shape)
            latcod=np.cos(locgeo[0])
            latcod2=np.cos(locgeo[1])

            lat1=latcod.view(1,-1,1)*xx
            lat2=latcod2.view(1,-1,1)*xx
            X=np.concatenate([X,lat1],axis=0)
            X=np.concatenate([X,lat2],axis=0)
        X[:,X[0] == 0] = np.nan
        if spread>0 and self.periodic_lon_expand:
            X=np.concatenate([X[:,:,-spread:],X,X[:,:,:spread]], axis=2)
        if self.same_padding:
            X,Y=self.pad_with_zero(X,0),self.pad_with_zero(Y,spread)
        X[X==0] = np.nan
        if self.outmasks[nd] is None:
            self.outmasks[nd] = self.compute_mask(X[:1])
        if self.inmasks[nd] is None:
            mask = X.copy()
            mask[mask==mask] = 1.
            self.inmasks[nd] = mask
        X[X!=X] = 0
        Y[Y!=Y] = 0
        return X,Y,self.outmasks[nd]

    def pad_with_zero(self,Y,spread,padding_val=0,centered=False):
        #p3d = (0, self.dimens[1]-2*spread-Y.shape[2], 0, self.dimens[0]-2*spread-Y.shape[1])
        if not centered:
            p3d = (0, self.dimens[1]-2*spread-Y.shape[2], 0, self.dimens[0]-2*spread-Y.shape[1])
        else:
            d1=self.dimens[1]-2*spread-Y.shape[2]
            d2=self.dimens[0]-2*spread-Y.shape[1]
            p3d = (d1//2,d1-d1//2 , d2//2, d2-d2//2)
        Y = F.pad(torch.from_numpy(Y), p3d, "constant", padding_val).numpy()
        return Y


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self,ds_data,domains,depthval,latfeat,temperature,lsrp,normalization :str = "absolute",division : int = 1):
        self.rec_field = 1
        self.normalization = normalization
        self.same_padding = False
        if 'st_ocean' in list(ds_data.coords.keys()):
            assert depthval>0
#             depth values are
#             [   5.03355 ,   55.853249,  110.096153,  181.312454,  330.007751,
#                       1497.56189 , 3508.633057]
            depthvals=ds_data.coords['st_ocean'].values
            ind = np.argmin(np.abs(depthvals - depthval))
            ds_data = ds_data.isel(st_ocean = ind)
        input_fields = ["usurf","vsurf"]
        output_fields = ["Su","Sv"]
        if temperature:
            input_fields.append("surface_temp")
            output_fields.append("ST")
        if lsrp:
            output_fields = [f"{outputf}_r" for outputf in output_fields] + output_fields

        self.input_fields = input_fields
        self.output_fields = output_fields

        self.latfeat = latfeat

        self.ds_data = ds_data.sel(yu_ocean=slice(-85, 85))
        self.domains=domains
        self.num_domains=len(self.domains['xmin'])
        tot_time=self.ds_data.time.shape[0]
        self.time_st=np.int64(np.floor(tot_time*self.domains['tmin']))
        self.time_tr=np.int64(np.ceil(tot_time*self.domains['tmax']))
        self.tot_time=self.time_tr-self.time_st
        self.num_time=np.int64(np.ceil(self.tot_time))

        dimens=[]
        for nd in range(self.num_domains):
            datsel=self.ds_data.sel(time=self.ds_data.time[0].data, \
                xu_ocean=slice(self.domains['xmin'][nd], self.domains['xmax'][nd]),\
                yu_ocean=slice(self.domains['ymin'][nd], self.domains['ymax'][nd]))
            dimens.append([datsel.yu_ocean.shape[0],datsel.xu_ocean.shape[0]])
        dimens=torch.tensor(dimens)
        self.dimens=torch.amax(dimens,dim=0)
        self.glbl_data=self.ds_data.yu_ocean.shape[0]*self.ds_data.xu_ocean.shape[0]==self.dimens[0]*self.dimens[1]

        self.periodic_lon_expand = self.glbl_data
        coords=[]
        icoords=[]
        geo=[]

        y=self.ds_data.yu_ocean.values
        x=self.ds_data.xu_ocean.values

        # lat1, lat2, lng1, lng2 = geographic_features2(len(y),net.spread*2+1)
        for nd in range(self.num_domains):
            datsel=self.ds_data.sel(time=self.ds_data.time[0].data,\
                xu_ocean=slice(self.domains['xmin'][nd], self.domains['xmax'][nd]),\
                yu_ocean=slice(self.domains['ymin'][nd], self.domains['ymax'][nd]))

            xx=datsel.xu_ocean.values
            yy=datsel.yu_ocean.values

            i0=np.argmin(np.abs(yy[0]-y))
            i1=np.argmin(np.abs(yy[-1]-y))+1

            j0=np.argmin(np.abs(xx[0]-x))
            j1=np.argmin(np.abs(xx[-1]-x))+1

            # locgeo = lat1[i0:i1], lat2[i0:i1], lng1[j0:j1], lng2[j0:j1]
            # locgeo = [torch.tensor(ll,dtype=torch.float32) for ll in locgeo]
            # geo.append(locgeo)
            coords.append([yy,xx])
            icoords.append([i0,i1,j0,j1])

        self.outmasks = [None]*self.num_domains
        self.inmasks = [None]*self.num_domains

        self.coords=coords
        # self.geo=geo
        self.icoords=icoords
        self.no_more_mask_flag=True

        self.inscalars = None
        self.outscalars = None
        y,x=self.coords[0]
        dy=y[1:]-y[:-1]
        dx=x[1:]-x[:-1]

        mdy=np.mean(dy)
        mdx=np.mean(dx)
        self.pooler=nn.MaxPool2d(self.rec_field,stride=1)
        self.box_km=[mdy*111,mdx*85]
    def load_geo(self,rec_field):
        y=self.ds_data.yu_ocean.values
        x=self.ds_data.xu_ocean.values
        lat1, lat2, lng1, lng2 = [None]*4#geographic_features2(len(y),rec_field)
        geo = []
        for nd in range(self.num_domains):
            datsel=self.ds_data.sel(time=self.ds_data.time[0].data,\
                xu_ocean=slice(self.domains['xmin'][nd], self.domains['xmax'][nd]),\
                yu_ocean=slice(self.domains['ymin'][nd], self.domains['ymax'][nd]))

            xx=datsel.xu_ocean.values
            yy=datsel.yu_ocean.values

            i0=np.argmin(np.abs(yy[0]-y))
            i1=np.argmin(np.abs(yy[-1]-y))+1

            j0=np.argmin(np.abs(xx[0]-x))
            j1=np.argmin(np.abs(xx[-1]-x))+1

            locgeo = lat1[i0:i1], lat2[i0:i1], lng1[j0:j1], lng2[j0:j1]
            locgeo = [torch.tensor(ll,dtype=torch.float32) for ll in locgeo]
            geo.append(locgeo)
        return geo
    def receive_scalars(self,dataset:'Dataset'):
        self.inscalars = dataset.inscalars
        self.outscalars = dataset.outscalars
        self.rec_field = dataset.rec_field
        self.pooler = dataset.pooler
        self.geo  = dataset.geo

    def compute_normalization_constants(self,):

        stats = {}
        for key in self.input_fields + self.output_fields:
            mom1 = 0
            mom2 = 0
            M=50
            for t in range(M):
                subds_data  = self.ds_data.isel(time = t)
                vals = subds_data[key].values
                vals[vals!=vals]= 0
                vals = vals[vals!=0]

                if self.normalization == "standard":
                    mom1 += np.mean(vals)
                    mom2 += np.mean(np.square(vals))
                elif self.normalization == "absolute":
                    mom1 += np.mean(vals)*0
                    mom2 += np.mean(np.abs(vals))

            mom1 = mom1/M
            mom2 = mom2/M
            if self.normalization == "standard":
                stats[key] = (mom1,np.sqrt(mom2 - np.square(mom1)))
            elif self.normalization == "absolute":
                stats[key] = (mom1,mom2)
        self.save_affine_maps(stats,)
        return stats
    def set_same_padding(self,flag):
        self.same_padding = flag
    def get_affine_maps(cls,st,inflds):
        avgscalars = [st[key][0] for key in inflds]
        stdscalars = [st[key][1] for key in inflds]
        return avgscalars,stdscalars

    def save_affine_maps(self,stats,):
        self.inscalars = self.get_affine_maps(stats,self.input_fields)
        self.outscalars = self.get_affine_maps(stats,self.output_fields)

    def __len__(self):
        'Denotes the total number of samples'
        return self.num_domains*self.num_time
    def compute_mask(self,x):
        prt = torch.from_numpy(np.stack([x],axis=0))
        prt[prt==prt] = 0.
        prt[prt!=prt] = 1.
        mask = 1.-self.pooler(prt).numpy()[0]
        return mask
    def land_density_map(self,dlat,dlon):

        X,_,_ = self[0]
        X = X[:1]
        X[X==0] = np.nan
        # X = X[:,::-1]
        mask = 1-self.compute_mask(X)
        conv  = nn.Conv2d(1,1,(dlat,dlon))
        conv.weight.data = conv.weight.data*0+1
        conv.bias.data = conv.bias.data*0
        mask = torch.from_numpy(np.stack([mask],axis=0)).type(torch.float32)
        density = conv(mask)/dlat/dlon
        # print('X.shape: ',X.shape)
        # print('mask.shape: ',mask.shape)
        # print('density.shape: ',density.shape)
        return density.detach().numpy()[0,0]
    def most_ocean_points(self,dy,dx,sep,num):
        density = self.land_density_map(dy,dx)
        # import matplotlib.pyplot as plt
        # plt.imshow(density[::-1])
        # plt.savefig('density.png')
        # plt.close()
        # if not np.any(density==0):
        #     return None
        inds = np.dstack(np.unravel_index(np.argsort(density.ravel()), density.shape))[0]
        # inds = inds[::-1,:]
        inds = inds + self.rec_field//2
        lat,lon = self.coords[0]
        spread = self.rec_field//2
        lon=np.concatenate([lon[-spread:],lon,lon[:spread]], axis=0)
        # print(lat.shape,lon.shape,density.shape)
        # print(lat[::10])
        ymin,xmin = lat[inds[:,0]],lon[inds[:,1]]
        ymax,xmax = lat[inds[:,0]+dy],lon[inds[:,1]+dx]
        # print(lat.shape,lon.shape)
        def remove_close_ones(vecs,s,i,d):
            vec = vecs[s]
            I = np.where(np.abs(vec[i]-vec[i+1:])>d)[0]
            if len(I)==0:
                return vecs,False
            I = I +i+1
            for j in range(len(vecs)):
                vec = vecs[j]
                vec = np.concatenate([vec[:i],vec[I]],axis=0)
                vecs[j] = vec
            return vecs,True
        exts = [xmin,xmax,ymin,ymax,inds]
        coords = []
        for i in range(num):
            xmin,xmax,ymin,ymax,inds = exts
            if len(xmin) <= i :
                break
            ind = tuple(inds[i].tolist())
            # print(ind)
            d = density[ind]
            # print(d,xmin[i],xmax[i],ymin[i],ymax[i])
            coords.append([inds[0],inds[1]])
            exts,flag = remove_close_ones(exts,0,i,sep)
            if not flag:
                exts,flag = remove_close_ones(exts,2,i,sep)
        return coords

    def set_receptive_field(self,rec_field):
        self.rec_field = int(rec_field)
        if self.glbl_data:
            self.dimens[1]+=(rec_field//2)*2
        self.pooler=nn.MaxPool2d(self.rec_field,stride=1)
        # self.set_same_padding(True)
        self.geo = self.load_geo(rec_field)
    def domain_index(self,index):
        return index%self.num_domains

    def normalize(self,vec,input=True):
        if input:
            scalars = self.inscalars
        else:
            scalars = self.outscalars
        for i in range(len(vec)):
            u = vec[i]
            vec[i] = (u - scalars[0][i])/scalars[1][i]
        return vec
    def denormalize(self,vec, input= True):
        if input:
            scalars = self.inscalars
        else:
            scalars = self.outscalars
        for i in range(len(vec)):
            u = vec[i]
            vec[i] = u*scalars[1][i] + scalars[0][i]
        return vec
    def to_physical(self,vec,input=True):
        if input:
            if not self.latfeat:
                return self.denormalize(vec,input=input)
            else:
                return self.denormalize(vec[:-2],input=input)
        else:
            return self.denormalize(vec,input=input)

    def get_pure_data(self,domid,timeid):
        index = timeid*self.num_domains + domid
        X,Y,mask = self[index]
        lon,lat = self.coords[domid]
        return X,Y,mask,lon,lat
    def confine(self,nd,xmin,xmax,ymin,ymax,):#index = False):
        # if index:
        #     datsel=self.ds_data.isel(time=self.ds_data.time[0].data, \
        #         xu_ocean=slice(xmin,xmax),\
        #         yu_ocean=slice(ymin,ymax))

        self.domains['xmin'][nd], self.domains['xmax'][nd],\
            self.domains['ymin'][nd], self.domains['ymax'][nd]\
                = xmin,xmax,ymin,ymax
        # lat,lon = self.coords[0]
        # print()
        dimens=[]
        for nd in range(self.num_domains):
            datsel=self.ds_data.sel(time=self.ds_data.time[0].data, \
                xu_ocean=slice(self.domains['xmin'][nd], self.domains['xmax'][nd]),\
                yu_ocean=slice(self.domains['ymin'][nd], self.domains['ymax'][nd]))
            dimens.append([datsel.yu_ocean.shape[0],datsel.xu_ocean.shape[0]])
        dimens=torch.tensor(dimens)
        self.dimens=torch.amax(dimens,dim=0)
        self.glbl_data=self.ds_data.yu_ocean.shape[0]*self.ds_data.xu_ocean.shape[0]==self.dimens[0]*self.dimens[1]
        self.periodic_lon_expand = self.glbl_data

        coords=[]
        icoords=[]

        y=self.ds_data.yu_ocean.values
        x=self.ds_data.xu_ocean.values
        for nd in range(self.num_domains):
            datsel=self.ds_data.sel(time=self.ds_data.time[0].data,\
                xu_ocean=slice(self.domains['xmin'][nd], self.domains['xmax'][nd]),\
                yu_ocean=slice(self.domains['ymin'][nd], self.domains['ymax'][nd]))

            xx=datsel.xu_ocean.values
            yy=datsel.yu_ocean.values

            i0=np.argmin(np.abs(yy[0]-y))
            i1=np.argmin(np.abs(yy[-1]-y))+1

            j0=np.argmin(np.abs(xx[0]-x))
            j1=np.argmin(np.abs(xx[-1]-x))+1

            coords.append([yy,xx])
            icoords.append([i0,i1,j0,j1])

        self.coords=coords
        self.icoords=icoords
        # print(icoords)


    def __getitem__(self,index):
        nd=index%self.num_domains
        nt=np.int64(np.floor(index/self.num_domains))+self.time_st
        datsel=self.ds_data.isel(time=nt).sel(xu_ocean=slice(self.domains['xmin'][nd], self.domains['xmax'][nd]), \
                                    yu_ocean=slice(self.domains['ymin'][nd], self.domains['ymax'][nd]))
        X=[datsel[key].values for key in self.input_fields]
        spread = self.rec_field//2
        # print(f'spread = {spread}')
        if spread > 0:
            # print(spread,datsel[self.output_fields[0]].shape)
            Y=[datsel[key].values[spread:-spread,:] for key in self.output_fields]
            # Y=[y[::-1] for y in Y]

            nouts = len(self.output_fields)
            if not self.periodic_lon_expand:
                Y=[Y[i][:,spread:-spread] for i in range(nouts)]
        else:
            Y=[datsel[key].values for key in self.output_fields]
            # Y=[y[::-1] for y in Y]

        if self.normalization is not None and self.inscalars is not None and self.outscalars is not None:
            X = self.normalize(X,input=True)
            Y = self.normalize(Y,input=False)
        X = np.stack(X,axis = 0)
        Y = np.stack(Y,axis = 0)

        if self.latfeat:
            locgeo=self.geo[nd]
            xx=np.ones(X[:1].shape)
            latcod=np.cos(locgeo[0])
            latcod2=np.cos(locgeo[1])

            lat1=latcod.view(1,-1,1)*xx
            lat2=latcod2.view(1,-1,1)*xx
            X=np.concatenate([X,lat1],axis=0)
            X=np.concatenate([X,lat2],axis=0)
        X[:,X[0] == 0] = np.nan
        if spread>0 and self.periodic_lon_expand:
            X=np.concatenate([X[:,:,-spread:],X,X[:,:,:spread]], axis=2)
        if self.same_padding:
            X,Y=self.pad_with_zero(X,0),self.pad_with_zero(Y,spread)
        X[X==0] = np.nan
        if self.outmasks[nd] is None:
            self.outmasks[nd] = self.compute_mask(X[:1])
        if self.inmasks[nd] is None:
            mask = X.copy()
            mask[mask==mask] = 1.
            self.inmasks[nd] = mask
        X[X!=X] = 0
        Y[Y!=Y] = 0
        return X,Y,self.outmasks[nd]

    def pad_with_zero(self,Y,spread,padding_val=0,centered=False):
        #p3d = (0, self.dimens[1]-2*spread-Y.shape[2], 0, self.dimens[0]-2*spread-Y.shape[1])
        if not centered:
            p3d = (0, self.dimens[1]-2*spread-Y.shape[2], 0, self.dimens[0]-2*spread-Y.shape[1])
        else:
            d1=self.dimens[1]-2*spread-Y.shape[2]
            d2=self.dimens[0]-2*spread-Y.shape[1]
            p3d = (d1//2,d1-d1//2 , d2//2, d2-d2//2)
        Y = F.pad(torch.from_numpy(Y), p3d, "constant", padding_val).numpy()
        return Y
