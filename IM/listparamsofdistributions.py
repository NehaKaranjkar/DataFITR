#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 12:06:49 2023

@author: lekshmi
"""

import scipy.stats as stats
import re,math,scipy
import pandas as pd
import numpy as np
import sys


def getcontinuousdist():
    continuous_all=[]
    all_dist = [getattr(stats, d) for d in dir(stats) if isinstance(getattr(stats, d), stats.rv_continuous)]
    filtered = [x for x in all_dist if ((x.a <= 0) & (x.b == math.inf))]
    filtered=all_dist
    pat = r's.[a-zA-Z0-9_]+_g'
    
    for i in filtered:
        s = str(i)
        #print(s)
        span=re.search(pat, s).span()
        dist=s[span[0]+2:span[1]-2]
        continuous_all.append(dist)
        
    for i in ['levy_stable','studentized_range','kstwo','skew_norm','vonmises','trapezoid','reciprocal']:
        continuous_all.remove(i)
    return continuous_all



def list_parameters(distribution):
    """List parameters for scipy.stats.distribution.
    # Arguments
        distribution: a string or scipy.stats distribution object.
    # Returns
        A list of distribution parameter strings.
    """
    if isinstance(distribution, str):
        distribution = getattr(stats, distribution)
    if distribution.shapes:
        parameters = [name.strip() for name in distribution.shapes.split(',')]
    else:
        parameters = []
    if distribution.name in scipy.stats._discrete_distns._distn_names:
        parameters += ['loc']
    elif distribution.name in scipy.stats._continuous_distns._distn_names:
        parameters += ['loc', 'scale']
    else:
        sys.exit("Distribution name not found in discrete or continuous lists.")
    return parameters



def dictionarize():
    distlist=getcontinuousdist() 
    
    paramlist={}
    for i in distlist:
        params=list_parameters(i)
        paramlist[i]=params
    return paramlist


def gencode(dist,typ,data):
    
    if typ=='discrete':
        params=calc_param(data,dist)
        op=gencodediscrete(dist, params)
    else:
        params=calc_param(data,dist)
        op=gencodecontinuous(dist, params)
    return op
    
        
        
def gencodediscrete(dist,params):
    if dist=='binom':
        str_lib="import numpy as np\n"
        str_p='n='+str(params[0])+"\n"+'p='+str(params[1])+"\nnum_datapoints=100\n"
        str_gen_code="data=np.random.binomial({},{},{})".format('n','p','num_datapoints')
        output=str_lib+str_p+str_gen_code
    elif dist=='poisson':
            str_lib="import numpy as np\n"
            str_p='lamda='+str(params[0])+"\nnum_datapoints=100\n"
            str_gen_code="data=np.random.poisson({},{})".format('lamda','num_datapoints')
            output=str_lib+str_p+str_gen_code
            
    elif dist=='geom':
        str_lib="import numpy as np\n"
        
        str_p='p='+str(params)+"\nnum_datapoints=100\n"
        str_gen_code="data=np.random.geometric({},{})".format('p','num_datapoints')
        output=str_lib+str_p+str_gen_code
        
    return output
def gencodekde(data,inpvar):
    #str_comment="#Change the 'data' to the approriate column name"
    str_lib="import numpy as np\nfrom scipy import stats\n"
    str_param="num_datapoints=100\n"
    str_data="input_data=df['"+str(inpvar)+"']\n"
    str_code="kernel=stats.gaussian_kde(input_data)\ndata=kernel.resample(num_points)"
    output=str_lib+str_data+str_param+str_code
    return output
    
        
def gencodecontinuous(dist,params):
    continuous_popular={'expon':['lamda'],'norm':['mean','variance'],'lognorm':['mu','sigma'],'triang':['mode','lowerlimit','upperlimit'],'uniform':['lowerlimit','upperlimit']}
    if dist == 'expon':
        str_lib="import scipy\n"
        str_p='loc='+str(params[0])+'\nlamda='+str(params[1])+"\nnum_datapoints=100\n"
        str_gen_code="data=scipy.stats.{}.rvs(loc={},scale={},size={})".format(dist,'loc','lamda','num_datapoints')
        output=str_lib+str_p+str_gen_code
        
    elif dist=='norm':
        str_lib="import scipy\n"
        str_p='mean='+str(params[0])+"\n"+'variance='+str(params[1])+"\nnum_datapoints=100\n"
        str_gen_code="data=scipy.stats.{}.rvs(loc={},scale={},size={})".format(dist,'mean','variance','num_datapoints')
        output=str_lib+str_p+str_gen_code
        
    elif dist=='norm':
        str_lib="import scipy\n"
        str_p='mean='+str(params[0])+"\n"+'variance='+str(params[1])+"\nnum_datapoints=100\n"
        str_gen_code="data=scipy.stats.{}.rvs(loc={},scale={},size={})".format(dist,'mean','variance','num_datapoints')
        output=str_lib+str_p+str_gen_code
    elif dist=='lognorm':
        str_lib="import scipy\n"
        str_p='mean='+str(params[0])+"\n"+'variance='+str(params[1])+"\nnum_datapoints=100\n"
        str_gen_code="data=scipy.stats.{}.rvs(loc={},scale={},size={})".format(dist,'mean','variance','num_datapoints')
        output=str_lib+str_p+str_gen_code
        
    elif dist=='triang':
        str_lib="import scipy\n"
        str_p='mode='+str(params[0])+"\n"+'lowerlimit='+str(params[1])+"\n"+'upperlimit='+str(params[2])+"\nnum_datapoints=100\n"
        str_gen_code="data=scipy.stats.{}.rvs(a={},loc={},scale={},size={})".format(dist,'mode','lowerlimit','upperlimit','num_datapoints')
        output=str_lib+str_p+str_gen_code
        
    elif dist=='uniform':
        str_lib="import scipy\n"
        str_p='lowerlimit='+str(params[0])+"\n"+'upperlimit='+str(params[1])+"\nnum_datapoints=100\n"
        str_gen_code="data=scipy.stats.{}.rvs(loc={},scale={},size={})".format(dist,'lowerlimit','upperlimit','num_datapoints')
        output=str_lib+str_p+str_gen_code  
   
        
    else:
        
        str_gen_code="data=scipy.stats.{}.rvs("
        argvals=params[:-2]
        str_params="args="+str(argvals)+"\nnum_datapoints=100\n"
            
        str_gen_code+='{},loc={},scale={},size={})'
            
        str_lib="import scipy\n"
        str_gen=str_gen_code.format(dist,"args",params[-2], params[-1],"num_datapoints")
        output=str_lib+str_params+str_gen
        
    return output
        


        
def calc_param(data,distribution):
     if distribution=='binom':
         
         mean=np.mean(data)
         
         binom_n=max(data)
         binom_param=mean/binom_n
         return (binom_n,binom_param,)
     elif distribution=='poisson':
         mean=np.mean(data)
         return (mean,min(data))
     elif distribution=='geom':
         p=len(data)/sum(data)
         return (p)
     
    
     else:
         k=getattr(stats,distribution)
         return k.fit(data)        
        
