import itertools
from transforms.lsrp import get_compressed_weights
import xarray as xr
import os
import numpy as np
import torch.nn as nn
import torch 

from transforms.subgrid_forcing import forward_difference
from utils.xarray import concat, fromtorch, totorch

class ConvolutionalLSRP:
    def __init__(self,sigma,clats,span):
        lsrp = get_compressed_weights(sigma,span)
        lspan = len(lsrp.latkernel)//2
        span = int(np.minimum(span,lspan))
        rfield = 2*span + 1
        def shrink_shift_span(lspan,span):
            lrfield = lspan*2+1
            z = np.zeros((lrfield,lrfield))
            for i,j in itertools.product(np.arange(-lspan,lspan+1),np.arange(-lspan,lspan+1)):
                if i >=-span and j>=-span and i <= span and j <= span:
                    z[i + lspan,j+ lspan] = 1
            z = z.reshape([-1])
            I = np.where(z>0)[0]
            return I
        
        lsrp = lsrp.interp(clat = clats)
        lsrp = lsrp.sel(latkernel = slice(-span,span),lonkernel = slice(-span,span))
        if len(lsrp.latkernel)//2 != span:
            print(len(lsrp.latkernel),span)
            raise Exception
        

        ncomp_dlon = len(lsrp.ncomp_dlon)
        ncomp_dlat = len(lsrp.ncomp_dlat)
        
        def weights2convolution(weights):
            weights = weights.reshape(-1,1,rfield,rfield)
            conv = nn.Conv2d(1,weights.shape[0],rfield,bias = False)
            conv.weight.data = torch.from_numpy(weights).type(torch.float32)
            return conv
        
        def get_id_conv():
            conv = nn.Conv2d(1,(2*span+1)**2,2*span+1,bias = False)
            conv.weight.data = 0*conv.weight.data
            for i,j in itertools.product(np.arange(rfield),np.arange(rfield)):
                k = i*rfield + j
                conv.weight.data[k,0,i,j] = 1
            return conv
        conv_id = get_id_conv()
        # print('lspan,span',lspan,span)
        I = shrink_shift_span(lspan,span)
        # print('len(I),',len(I),)
        conv_dlat = weights2convolution(lsrp.weights_dlat.values[:,I,:,:])
        conv_dlon = weights2convolution(lsrp.weights_dlon.values[:,I,:,:])
        lat_transfer_dlat = lsrp.latitude_transfer_dlat.values
        lat_transfer_dlon = lsrp.latitude_transfer_dlon.values

        y = lat_transfer_dlat
        y = y[span:-span,:].transpose()
        y = np.stack([y],axis = 2)

        self.lat_transfer_dlat = torch.from_numpy(y).type(torch.float32)

        y = lat_transfer_dlon
        y = y[span:-span,:].transpose()
        y = np.stack([y],axis = 2)

        self.lat_transfer_dlon = torch.from_numpy(y).type(torch.float32)
        

        # if torch.cuda.is_available():  
        #     device = "cuda:0" 
        # else:  
        #     device = "cpu"  
        # self.device = device

        self.conv_dlat = conv_dlat#.to(device)
        self.conv_dlon = conv_dlon#.to(device)
        self.conv_shift = conv_id#.to(device)
        
        self.ncomp_dlat = ncomp_dlat
        self.ncomp_dlon = ncomp_dlon
        self.span = span
        self.slice = slice(span,-span)
        self.rfield = rfield
        self.clat = clats
    def single_latitude_weights(self,lat):
        def get_weights(conv_dlat,lat_transfer_dlat):
            i = np.argmin(np.abs(self.clat - lat))
            latdlat = torch.stack([lat_transfer_dlat[:,i:i+1]],dim = 3)
            shp = list(conv_dlat.weight.shape)
            shp[1] = int(shp[0]/latdlat.shape[0])
            shp[0] = latdlat.shape[0]
            dlatw = conv_dlat.weight.data.reshape(*shp)
            w = torch.sum(dlatw*latdlat,dim = 0)
            w = w.reshape([self.rfield]*4)
            return w
        dlatw = get_weights(self.conv_dlat,self.lat_transfer_dlat)
        dlonw = get_weights(self.conv_dlon,self.lat_transfer_dlon)
        return dlatw.numpy(),dlonw.numpy()
    def read_coords(self,u):
        if 'clat' in u.coords:
            return u.clat.values,u.clon.values
        else:
            return u.lat.values,u.lon.values

    def forward(self,u:xr.DataArray,v:xr.DataArray,T:xr.DataArray)->xr.Dataset:
        var = {}
        for a,name in zip((u,v,T),"u v T".split()):
            var[name] =  self._forward(u,v,a)
        return concat(**var)
    def _forward(self,u:xr.DataArray,v:xr.DataArray,T:xr.DataArray)->xr.DataArray:
        lat,lon = self.read_coords(u)        
        dTdx = forward_difference(T,'lon')
        dTdy = forward_difference(T,'lat')
        u,v,T,dTdy,dTdx = totorch(u,v,T,dTdy,dTdx,leave_nan = False)
        # print('T.shape',T.shape)
        with torch.no_grad():
            Tdlat = self.conv_dlat(T)
            Tdlon = self.conv_dlon(T)
        # print('Tdlat.shape',Tdlat.shape)
        Tdlat = Tdlat.reshape(self.ncomp_dlat,self.rfield**2,*Tdlat.shape[-2:])
        Tdlon = Tdlon.reshape(self.ncomp_dlon,self.rfield**2,*Tdlon.shape[-2:])
        with torch.no_grad():
            Slon = torch.sum(self.conv_shift(u)*Tdlon,dim = 1)
            Slat =  torch.sum(self.conv_shift(v)*Tdlat,dim = 1)
        Slat = torch.sum(Slat * self.lat_transfer_dlat, dim = 0)
        Slon = torch.sum(Slon * self.lat_transfer_dlon, dim = 0)
        Slres = u*dTdx + v*dTdy
        Slres = Slres.reshape(*Slres.shape[-2:])
        Slres = Slres[self.slice,self.slice]
        S = Slon + Slat + Slres
        S = fromtorch(S,lat,lon)
        return S
