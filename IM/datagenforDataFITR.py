# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:53:55 2023

@author: User
"""

import scipy,random
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import numpy as np




def genrvs(d1,d2,param1,param2,size1,size2):
    k=getattr(scipy.stats, d1)
    args=param1[:-2]
    k1=k.rvs(loc=param1[-2],scale=param1[-1],*args,size=size1)
    k=getattr(scipy.stats, d2)
    args=param2[:-2]
    k2=k.rvs(loc=param2[-2],scale=param1[-1],*args,size=size2)
    return np.concatenate((k1,k2))

def normal(N):
    return scipy.stats.norm.rvs(random.randint(5,30),random.randint(0,2)+random.random(),size=N)
def expon(N):
    return scipy.stats.expon.rvs(random.random(), size=N)
def uniform(N):
    return scipy.stats.uniform.rvs(random.randint(5,30),random.randint(5,30), size=N)
def poisson(N):
    return np.random.poisson(random.random(),N)

def MVGaussian(N):
    np.random.seed(0)
    n_sample = N
    d_sample = 5
    cov_sample = np.eye(d_sample) + np.random.rand(d_sample, d_sample)+5
    sim_cov = cov_sample.transpose().dot(cov_sample)
    data = np.random.exponential(size=(n_sample, d_sample)) + np.random.multivariate_normal(
        np.zeros(d_sample), sim_cov, size=n_sample)
    return data


def multivariate_Gaussian_sample(N):
    data=MVGaussian(N)
    processdataMV={"temp":data[:,0],"pressure":data[:,1],"jobarrival_Machine":data[:,2],"usedrawmaterial_amount":data[:,3],"flow":data[:,4]}
    processdfMV=pd.DataFrame(processdataMV)
    #processdf.to_csv("sampledatamultivariate.csv")
    return processdfMV

def univariate_sample(N):
    processdata={"temp":normal(N),"pressure":uniform(N),"jobarrival_Machine":poisson(N),"usedrawmaterial_amount":expon(N),"timetaken_transport":normal(N)}
    processdf=pd.DataFrame(processdata)
    #processdf.to_csv("sampledata.csv")
    return processdf

                        