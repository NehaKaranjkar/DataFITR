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

def genrvs_old(N):
    distributionlist=[i+1 for i in range(5)]
    d1=random.choice(distributionlist)
    if N%2==0:
        N1=N//2
        N2=N//2
    else:
        N1=N//2
        N2=(N//2)+1
        
    #print(d1)
    if d1==1:
        k1=scipy.stats.expon.rvs(random.random(), size=N1)
        k2=scipy.stats.norm.rvs(random.randint(5,10),random.randint(1,2)+random.random(),size=N2)
    elif d1==2:
        k1=scipy.stats.expon.rvs(random.randint(1,2)+random.random(), size=N1)
        k2=scipy.stats.norm.rvs(random.randint(7,9),random.randint(0,2)+random.random(),size=N2)
    elif d1==3:
        k1=scipy.stats.norm.rvs(random.randint(5,10),random.randint(2,3)+random.random(),size=N1)
        k2=scipy.stats.norm.rvs(random.randint(13,15),random.randint(0,2)+random.random(),size=N2)
    elif d1==4:
        k1=scipy.stats.norm.rvs(random.randint(38,42),random.randint(0,2)+random.random(),size=N1)
        k2=scipy.stats.uniform.rvs(random.randint(40,43),random.randint(55,65), size=N2)
    else:
        k1=scipy.stats.uniform.rvs(random.randint(45,49),random.randint(60,65), size=N1)
        k2=scipy.stats.expon.rvs(random.randint(44,46)+random.random(), size=N2)
   
    
    #plt.hist(k1,bins=40)    
    #plt.hist(k2,bins=40)  
    #plt.show()
   
    return np.concatenate((k1,k2))


def genrvs(d1,d2,param1,param2,size1,size2):
    k=getattr(scipy.stats, d1)
    args=param1[:-2]
    k1=k.rvs(loc=param1[-2],scale=param1[-1],*args,size=size1)
    k=getattr(scipy.stats, d2)
    args=param2[:-2]
    k2=k.rvs(loc=param2[-2],scale=param1[-1],*args,size=size2)
    return np.concatenate((k1,k2))

def mixeddistribution(N):
    if N%2==0:
        N1=N//2
        N2=N//2
    else:
        N1=N//2
        N2=(N//2)+1
    data1= genrvs("norm","expon",[2,0.5],[1,0.5],N1,N2)
    data2= genrvs("norm","triang",[2,1],[0.5,0.5,1],N1,N2)
    data3= genrvs("norm","uniform",[2.5,0.06],[4,3],N1,N2)
    data4= genrvs("uniform","uniform",[6,6],[6,7],N1,N2)
    return random.choice([data1,data2,data3,data4])



def normal(N):
    return scipy.stats.norm.rvs(random.randint(5,30),random.randint(0,2)+random.random(),size=N)
def expon(N):
    return scipy.stats.expon.rvs(random.random(), size=N)
def uniform(N):
    return scipy.stats.uniform.rvs(random.randint(5,30),random.randint(5,30), size=N)
def poisson(N):
    return np.random.poisson(random.random(),N)
def geometric(N):
    return np.random.geometric(random.random(), size=N)




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
    processdata={"temp":normal(N),"pressure":uniform(N),"jobarrival_Machine":poisson(N),"usedrawmaterial_amount":expon(N),"timetaken_transport":geometric(N),"energy":mixeddistribution(N)}
    print(processdata)
    processdf=pd.DataFrame(processdata)
    processdf.to_csv("sampledatajun13.csv")
    return processdf



                        