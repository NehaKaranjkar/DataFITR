#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from IM import listparamsofdistributions



def generatediscrete(dist):
    
    if dist=='binom':
        k=st.text_input("Enter the params separated by comas, n,p")
        
        if k:
            lis=k.split(",")
            params=[float(i) for i in lis]
            str_lib="import numpy as np\n"
            str_p='n='+str(params[0])+"\n"+'p='+str(params[1])+"\nnum_datapoints=100\n"
            str_gen_code="data=np.random.binomial({},{},{})".format('n','p','num_datapoints')
            output=str_lib+str_p+str_gen_code
            st.text("Adjust the num_datapoints parameter to change the number of points to generate")
            st.code(output)
    
    elif dist=='poisson':
        k=st.text_input("Enter the params separated by comas, lambda")
        if k:
            lis=k.split(",")
            params=[float(i) for i in lis]
            
            str_lib="import numpy as np\n"
            str_p='lamda='+str(params[0])+"\nnum_datapoints=100\n"
            str_gen_code="data=np.random.poisson({},{})".format('lamda','num_datapoints')
            output=str_lib+str_p+str_gen_code
            st.text("Adjust the num_datapoints parameter to change the number of points to generate")
            st.code(output)
            
    elif dist=='geom':
        k=st.text_input("Enter the params separated by comas, p")
        if k:
            lis=k.split(",")
            params=[float(i) for i in lis]
            str_lib="import numpy as np\n"
            str_p='p='+str(params[0])+"\nnum_datapoints=100\n"
            str_gen_code="data=np.random.geometric({},{})".format('p','num_datapoints')
            output=str_lib+str_p+str_gen_code
            st.text("Adjust the num_datapoints parameter to change the number of points to generate")
            st.code(output)
  


def generate(dist):
    
    k=getattr(stats,dist) 
    data=np.random.normal(1,10,100)# just some dummy data to find the number of params.
    numparams=len(k.fit(data))
    st.write("{} parameters are required to generate this distribution including size. Please follow this format loc,scale,c1,c2,..,size".format(numparams+1))
    if numparams>2:

        k=st.text_input("Enter the params separated by comas, loc,scale,c,size")
        
        if k:
            lis=k.split(",")
            paramlist=[float(i) for i in lis]
        
                
            
            str_lib="import scipy\n"
            
            args='args={}\n'.format(paramlist[2:-1])
              
            str_gen='scipy.stats.{}.rvs(*args,loc={},scale={},size={})'.format(dist,paramlist[0],paramlist[1],int(paramlist[-1]))
            output=str_lib+args+str_gen
            st.code(output)
            #st.text_area(label="Code to generate the data",value=output)
    else:

        k=st.text_input("Enter the params separated by comas, loc,scale,size")
        if k:
            lis=k.split(",")
            paramlist=[float(i) for i in lis]
            
        
            
            str_lib="import scipy\n"
              
            str_gen='scipy.stats.{}.rvs(loc={},scale={},size={})'.format(dist,paramlist[0], paramlist[-2],int(paramlist[-1]))
            output=str_lib+str_gen
            st.code(output)
            
st.subheader("Random Variate Generation")
st.markdown("---")    
st.sidebar.header("Random Variate Generation")
#st.sidebar.image("/home/lekshmi/Downloads/DataFitter/MISC/datafitrlogo.PNG", use_column_width=True)   
st.sidebar.image("./MISC/datafitrlogo.PNG", use_column_width=True)     
            


datatype=st.selectbox('Choose the type of datatype', ['real valued','integer valued'])
if datatype=='real valued':
    distlist=listparamsofdistributions.getcontinuousdist()
    dist = st.selectbox("Choose a distribution to proceed", distlist)
    generate(dist)

else:
    dist = st.selectbox("Choose a distribution to proceed", ['geom','binom','poisson'])
    generatediscrete(dist)
    
    
    
    
   
        
        