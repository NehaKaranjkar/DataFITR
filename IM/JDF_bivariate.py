# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 00:43:05 2023

@author: leksh
"""
import pandas as pd
import matplotlib.pyplot as plt


def cartesianProduct(set_a, set_b):
    result =[]
    for i in range(0, len(set_a)):
        for j in range(0, len(set_b)):
 
            # for handling case having cartesian
            # product first time of two sets
            if type(set_a[i]) != list:         
                set_a[i] = [set_a[i]]
                 
            # copying all the members
            # of set_a to temp
            temp = [num for num in set_a[i]]
             
            # add member of set_b to 
            # temp to have cartesian product     
            temp.append(set_b[j])             
            result.append(temp)
    return result

def Cartesian(list_a):
     
    # result of cartesian product
    # of all the sets taken two at a time
    temp = list_a[0]
    n=len(list_a) 
    # do product of N sets 
    for i in range(1, n):
        temp = cartesianProduct(temp, list_a[i])
    #print("kkkkk",temp)    
    return temp


import pandas as pd


def to_JDF(df,bins=50):

    #df=data[columns]

    dic={}
    for ind in range(len(df)):
        dic[ind]=tuple(df.iloc[ind])
        
    dic_ind={}
    dic_bins={}
    
    k=[]
    for col in df.columns:
        ax = plt.hist(df[col],bins=bins)
        plt.close()
        dic_bins[col]=ax[1]
        dic_ind[col]=list(range(1,bins+1))
        #print(dic_ind)
        k.append(dic_ind[col])
    #print("here",dic_ind)
    copula_list=Cartesian(k)
    #print(dic_ind)
    dic_copula={}
    for i in copula_list:
        ind1=i[0]
        ind2=i[1]
        for row in df.index:
            if df.iloc[row][0]<dic_bins[df.columns[0]][ind1] and df.iloc[row][0]>=dic_bins[df.columns[0]][ind1-1]:
                if df.iloc[row][1]<dic_bins[df.columns[1]][ind2] and df.iloc[row][1]>=dic_bins[df.columns[1]][ind2-1]:
                    kk=tuple(i)
                    if kk in dic_copula:
                        dic_copula[kk]+=1
                    else:
                        dic_copula[kk]=1
    return dic_copula,dic_bins
                    

                    
                
                
#importing the libraries

def plot_jdf(dic_copula,dic_bins):
    x=[]
    y=[]
    cols=list(dic_bins.keys())
    for i in dic_copula.keys():
        lower=i[0]-1
        upper=i[1]-1
        x.append(dic_bins[cols[0]][lower])
        y.append(dic_bins[cols[1]][upper])
        
    
            
    z=[i/sum(dic_copula.values()) for i in dic_copula.values()]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    #print(x,y,z)
    ax1.scatter3D(x,y, z, color="red")
    
    
    
    plt.show()
       
   
    
    
    
