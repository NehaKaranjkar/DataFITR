#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 19:15:16 2023

@author: lekshmi
"""

import sys,io,os
import streamlit as st
import pandas as pd
import scipy.stats as stats,time
from io import StringIO
#sys.path.append('/home/lekshmi/Downloads/my_app/IM') 
sys.path.append('./IM') 
#import modular_IM
from IM import modular_IM
from IM import kde_silverman_both
#import kde_silverman_both
#fromm IM import wrapper
import math
import re
from IM import listparamsofdistributions
import numpy as np
#from statsmodels.tsastattools import adfuller
import seaborn as sns
from scipy.stats import pearsonr #to calculate correlation coefficient
import matplotlib.pyplot as plt
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb


#st.title("This part will be added soon")

def my_function(data,distlist,distributions,typ='continuous',dist='my1 distribution',bins=100,gof="ks",):
    
    #st.write("Please wait while I fit your data")
    #modelGUI=modular_IM.modelmatch()
    if len(np.unique(np.array(data)))==1:
        #st.write("chakkakakakkaka")
        restyp='constant'
        result=list()
        plotdata="nill"
        pval=data[0]
        return  restyp,result, plotdata,pval
    #st.write("hdhhdhdhdhdh")
    modelGUI=modular_IM.modelmatch(data ,typ,dist,bins,gof,distlist,distributions)
    #st.write(modelGUI.data)
    plotdata,pval=modelGUI.bestfit(distlist,distributions)
    GOFresult,SSEresult=modelGUI.printresult()
    #st.write(pd.DataFrame(plotdata))
    GOF=pd.DataFrame(GOFresult).T
    SSE=pd.DataFrame(SSEresult).T
    
    restyp='nonconstant'
    GOF.rename({0: GOF.iloc[0][0], 1: GOF.iloc[0][1]}, axis=1, inplace=True)
    GOF.drop('Test',axis=0,inplace=True)
    cols=list(GOF.columns)
    for i in cols:
        GOF[i] = GOF[i].apply(lambda X: X[0])
        
    
    SSE.rename({0: SSE.iloc[0][0]}, axis=1, inplace=True)
    SSE.drop('Test',axis=0,inplace=True)
    result = pd.merge(GOF, SSE, on=GOF.index)
    result.rename({'key_0': 'Test'}, axis=1, inplace=True)
    #st.write(plotdata)
    
    return restyp,result,plotdata,pval

def pearsonCorr(x, y, **kws): 
    (r, _) = pearsonr(x, y) #returns Pearson’s correlation coefficient, 2-tailed p-value)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),xy=(.7, .9), xycoords=ax.transAxes)
    
@st.experimental_memo
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


def covariancetostring(cov):
    str_covprint="["
    for rows in cov:
        str_covprint+="["
        for i in rows:
            str_covprint+=str(i)
            str_covprint+=","
        str_covprint+="],"
    str_covprint+="]"
    return str_covprint

def numcolerror():
    st.info("There is only one data column in the uploaded file. Not enough features to find the correlation. Please upload a file with atleast two features")
    


st.set_page_config(page_title="DataFITR",page_icon="📈",layout="wide")

st.sidebar.subheader("Modeling of multivariate gaussian data")

#st.sidebar.image("/home/lekshmi/Downloads/DataFitter/MISC/datafitrlogo.PNG", use_column_width=True)
st.sidebar.image("./MISC/datafitrlogo.PNG", use_column_width=True)
st.subheader("Modeling of multivariate gaussian data")
st.markdown("---")



data_col,button_col=st.columns(2)
with data_col:
    

    st.markdown('''Please upload a CSV file with the variables to fit and proceed.''')
    uploaded_file = st.file_uploader("Choose a file. Please ensure that the column names have no space character in it.")
with button_col:
    st.markdown('''If you do not have a dataset and are curious to know how this tool works, download the CSV file and proceed.                                                                                                                                                
                ''')
    
    df = pd.read_csv("./MISC/sampledata.csv")
    csv = convert_df(df)
    
    st.download_button(
       "Download sample data",
       csv,
       "sampledata.csv",
       "text/csv",
       key='download-csv',
    )




#"st.session_state_obj",st.session_state
if uploaded_file is not None:
    st.header("Data Exploration")
      
    df = pd.read_csv(uploaded_file)
    if len(df.columns)==1:
        numcolerror()
    else:
        filenamevalue=uploaded_file.name
        folder_name=filenamevalue.split()[0][:-4]#removing .csv from name
        
        fig= plt.figure(figsize=(15, 15))
        j=1
        cols=list(df.columns)
        
        st.subheader("Marginal distribution of the columns in the dataset")
        for i in cols:
            plt.subplot(len(cols),3,j)
            plt.tight_layout()
            a=sns.distplot(df[i],bins=100,rug=True)
            a.set_xlabel(i,fontsize=15)
            a.set_ylabel("density",fontsize=15)
            j=j+1
        st.pyplot(fig)
    
        cols=list(df.columns)
        st.header("Detecting Correlation")
        
        corrfull_col,corrthreshold_col=st.columns(2)
        with corrfull_col:
            correlation=df.corr()
            
            #st.subheader('Correlation between every pair of columns in the dataset uploaded')
            st.markdown("""<style>.big-font {font-size:20px !important;}</style>""", unsafe_allow_html=True)
            st.markdown('<p class="big-font">Correlation between every pair of columns in the dataset uploaded', unsafe_allow_html=True)
            
            #st.write(correlation)
            
    
            # To check the correlation coefficient between variables
            #print(corr)
            fig1=plt.figure(figsize=(10,10))
            a = sns.heatmap(correlation, annot=True, fmt='.3f',annot_kws={'size':16})
            rotx = a.set_xticklabels(a.get_xticklabels(), rotation=25,size=13)
            roty = a.set_yticklabels(a.get_yticklabels(), rotation=0,size=15)
            st.pyplot(fig1)
    
        with corrthreshold_col:
            corr_triupper=correlation.where(~np.tril(np.ones(correlation.shape)).astype(np.bool))
            corr_triupper=corr_triupper.stack()
            #st.caption(":red[Columns with a pearson correlation value greater than 0.5 and their respective correlation value]")
            st.markdown('<p class="big-font">Columns with a pearson correlation value greater than 0.5 and their respective correlation value', unsafe_allow_html=True)
           
            #st.write(corr_triupper[corr_triupper>0.5] )   
            outputdframe=pd.DataFrame(corr_triupper[corr_triupper>0.5])
            #outputdframe=outputdframe.reset_index(drop=True)
            th_props = [
              ('font-size', '14px'),
              ('text-align', 'center'),
              ('font-weight', 'bold'),
              ('color', '#6d6d6d'),
              ('background-color', '#f7ffff')
              ]
                                           
            td_props = [
              ('font-size', '14px')
              ]
                                             
            styles = [
              dict(selector="th", props=th_props),
              dict(selector="td", props=td_props)
              ]
            
            # table
            df2=outputdframe.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles)
            st.table(df2)
            corrvalues=list(corr_triupper[corr_triupper>0.5])
    
            dfcols=list(df.columns)
            corrlist=[]
            for i in dfcols:
                for corrval in corrvalues:
                    if corrval in correlation[i].values:
                        corrlist.append(i)
        #st.write(corrlist)
        multivars_inp=st.multiselect("Select the variables with correlation", corrlist,key='multivarcols')
        continuous_popular=['expon','norm','lognorm','triang','uniform','weibull_min','gamma']
        discrete_popular=['binom','poisson','geom']
        bins=100
        goodnessoffit='KStest'
        countnorm=0
        for inpvar in multivars_inp:
            if df[inpvar].dtype=='int64':
                distlist='discrete'
                distributions=[discrete_popular]
                datatyp='discrete'
            else:
                distlist='continuous'
                distributions=[continuous_popular]
                datatyp='continuous'
            dataname=inpvar   
            restyp,finresult,plotdata,pval=my_function(df[inpvar],distlist,distributions,datatyp,dataname,bins,'ks',)
            if restyp=='nonconstant':
                result_df = finresult.sort_values(by = goodnessoffit)
                up_df=result_df.copy()
                #st.write(up_df)
                dftoprint=result_df.head(10)
                df_gof_large=dftoprint.reset_index(drop=True)
                #st.write(df_gof_large.head(1))
                if df_gof_large['Test'][0]=='norm':
                    countnorm+=1
        if countnorm!=len(multivars_inp):
            st.warning("The univariate marginal distribution of all or part of the selected columns are not normal distributions. So the dependence between the selected columns may not be Gaussian")
        
    
       
    
        fulllist=[]
        for i in range(len(df[dfcols[0]])):
            inlist=[]
            for corr in multivars_inp:
                #st.write(corr)
                inlist.append(df[corr][i])
                
            fulllist.append(inlist)
        
        fit_mult=st.checkbox("Generate multivariate IID data.")
       
        if fit_mult: 
            st.write("Please wait while I fit your data")
            
            cov_col,gendata_col=st.columns(2)
            with cov_col:
                cov=np.cov(fulllist, rowvar=False)
                #st.write(fulllist)
                st.markdown('<p class="big-font">Covariance matrix of the selected columns', unsafe_allow_html=True)
                #st.caption("Covariance matrix of the selected columns")
                st.write(cov)
            with gendata_col:
                st.markdown('<p class="small-font">Code to generate the multivariate Gaussian distribution', unsafe_allow_html=True)
                d=len(multivars_inp)
                m=[np.mean(df[multivars_inp[i]]) for i in range(d)]
                str_gen_code="data=multivariate_normal.rvs(mean="
                mean=str(m)
                
               
                str_cov=covariancetostring(cov)
                
                str_params="mean="+str(m)+"\ncov="+str_cov+"\nnum_datapoints=100\n"
                    
                str_gen_code+='{},cov={},size={})'
                    
                str_lib="from scipy.stats import multivariate_normal\n"
                  
                str_gen=str_gen_code.format("mean","cov","num_datapoints")
                output=str_lib+str_params+str_gen
                st.markdown('<p class="small-font">Adjust the num_datapoints parameter to change the number of points to generate', unsafe_allow_html=True)
    
                st.code(output)
            
        
    