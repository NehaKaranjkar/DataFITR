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
from IM import datagenforDataFITR
from IM import kde_silverman_both
#import kde_silverman_both
#fromm IM import wrapper
import math
#import plotly.express as px
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
def to_csv(df):
    data_csv = df.to_csv(path_or_buf=None, sep=',', index=False) 
    return data_csv


def numcolerror():
    st.info("There is only one data column in the uploaded file. Not enough features to find the correlation. Please upload a file with atleast two features")
    

def main():
    st.set_page_config(page_title="DataFITR",page_icon="📈",layout="wide")
    
    st.sidebar.subheader("Fitting Gaussian Multivariate Data")
    
    #st.sidebar.image("/home/lekshmi/Downloads/DataFitter/MISC/datafitrlogo.PNG", use_column_width=True)
    st.sidebar.image("./MISC/datafitrlogo.PNG", use_column_width=True)
    st.subheader("Fitting Gaussian Multivariate Data")
    st.markdown("---")
    
    
    
    data_col,button_col=st.columns(2)
    
    
    
    
    data_col,button_col=st.columns(2)
    with data_col:
        
        datainput_formMVG=st.radio("Select a data source",
                ('Upload a CSV','Use a sample synthetic data',))
        if datainput_formMVG=='Use a sample synthetic data':
            numdatagen=st.slider("Enter the number of datapoints to generate", 50, 5000,2000)
       
        else:
            st.write("The expected format is as shown below. ")
            st.image("./MISC/sampledata.PNG",width=500)
            st.markdown("""
            The tool expects the data to be in a csv format where the column name is the name of the process variable and the rows are the observations.
            """)
    with button_col:
        if datainput_formMVG=='Upload a CSV':
            try:
                
                st.markdown('''Please upload a CSV file with the variables to fit and proceed.''')
                uploaded_file = st.file_uploader("Choose a file.",type="csv")
                #st.session_state.data_generatedMVG=False
                if uploaded_file is None:
                    st.warning("Please upload a csv file to proceed")
                    return
                    
                        
                else:
                    dfMVG = pd.read_csv(uploaded_file)
                    st.session_state.data_generatedMVG=True
                    st.session_state.dataMVG=dfMVG
                    startcheck=dfMVG.describe(include="all")
                    
            except:
                st.warning("Please upload a csv file which adheres to the format mentioned in the documentation.")
                #uploaded_file = st.file_uploader("Choose a file.")
                #IM_uni(dfMVG)    
                
        
            
            #IM_uni(dfMVG)
    
        else:
            
            #st.session_state.data_generatedMVG=False
            if st.button(("Regenerate a sample synthetic data" if "data_generatedMVG" in st.session_state else "Generate a sample synthetic data")):
                
                dfMVG=datagenforDataFITR.multivariate_Gaussian_sample(numdatagen)
                st.session_state.data_generatedMVG=True
                st.session_state.dataMVG=dfMVG
        
                with st.expander("View raw data"):
                    st.write(dfMVG)
                st.download_button('Download generated data as a CSV file', to_csv(dfMVG), 'sample_multivariate_gaussian_data.csv', 'text/csv')
                
    if  'data_generatedMVG' in st.session_state:
        #st.write(st.session_state.data_generated)
        
        if st.button("Start Exploratory Data Analysis"):
            st.session_state.button_clickedMVG=True
    if 'button_clickedMVG' in st.session_state:
            datatofitMVG=st.session_state['dataMVG']
            #st.write(datatofit)    
            IM_MVG(datatofitMVG)
            
      
    
    
    
    
    
    
def IM_MVG(df):
    #"st.session_state_obj",st.session_state
    
    st.header("Data Exploration")
      
    #df = pd.read_csv(uploaded_file)
    if len(df.columns)==1:
        numcolerror()
    else:
        #filenamevalue=uploaded_file.name
        #folder_name=filenamevalue.split()[0][:-4]#removing .csv from name
        folder_name='SampleMVG_CSVfile'
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
            corr_triupper=correlation.where(~np.tril(np.ones(correlation.shape)).astype(bool))
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
        corrlist=set(corrlist)
        
        
        if len(corrlist)>2:
            st.write("Select the components to visualize a 3D scatter plot")
            col1,col2,col3=st.columns(3)
            with col1:
                selected_x_axis=st.selectbox("x-axis", corrlist)
            with col2:
                selected_y_axis=st.selectbox("y-axis", [a for a in corrlist if (a!=selected_x_axis)])
            with col3:
                selected_z_axis=st.selectbox("z-axis", [a for a in corrlist if (a!=selected_x_axis and a!=selected_y_axis)])
                
             
            fig = plt.figure(figsize = (10, 7))
            ax = plt.axes(projection ="3d")
             #
            # Creating plot
            ax.scatter3D(df[selected_x_axis], df[selected_y_axis],df[selected_z_axis], color = "blue")
            ax.set_xlabel(selected_x_axis, fontweight ='bold')
            ax.set_ylabel(selected_x_axis, fontweight ='bold')
            ax.set_zlabel(selected_x_axis, fontweight ='bold')
            plt.title("3D scatter plot")
            st.pyplot(fig)
            # show plot
              
            
        
            
            #st.plotly_chart(fig, use_container_width=True)
        elif len(corrlist)==2:
            st.subheader("Select the components to visualize a 2D scatter plot")
            col1,col2=st.columns(2)
            with col1:
                selected_x_axis=st.selectbox("x-axis", corrlist)
           
                selected_y_axis=st.selectbox("y-axis", [a for a in corrlist if (a!=selected_x_axis)])
                
                  
                    
                 
                    
            with col2:
                fig = plt.scatter(df[selected_x_axis], df[selected_y_axis],color='blue')
               
                #fig.update_layout(scene = dict(aspectmode='cube'),template='plotly',margin={"l":0,"r":0,"t":0,"b":0},title_font_family="Times New Roman",title_font_color="red" )
                #fig.update_traces(marker_size=st.session_state.marker_size)
                
                ax.set_xlabel(selected_x_axis, fontweight ='bold')
                ax.set_ylabel(selected_x_axis, fontweight ='bold')
                
                plt.title("2D scatter plot")
                st.pyplot(fig)
                # show plot
                
        
        else:
            st.warning("No columnns with correlation. Please upload a file with variables with corelation")
            return
        
        
        multivars_inp=st.multiselect("Select the variables with correlation to display GOF values", corrlist,key='multivarcols')
        continuous_popular=['expon','norm','lognorm','triang','uniform','weibull_min','gamma']
        discrete_popular=['binom','poisson','geom']
        bins=100
        goodnessoffit='KStest'
        
        dfresultMVG=pd.DataFrame({'Test':[],'KStest':[],'Chi squared Test':[],'mean':[],'variance':[],'SSE':[],'column':[]})
        if multivars_inp:
            for inpvar in multivars_inp:
                
                distlist='continuous'
                distributions=[['norm']]
                datatyp='continuous'
                dataname=inpvar   
                restyp,finresult,plotdata,pval=my_function(df[inpvar],distlist,distributions,datatyp,dataname,bins,'ks',)
                if restyp=='nonconstant':
                    result_df = finresult.sort_values(by = goodnessoffit)
                    up_df=result_df.copy()
                    
                    #st.write(up_df)
                    dftoprint=result_df.head(10)
                    dftoprint['column']=inpvar
                    dftoprint['mean']=df[inpvar].mean()
                    dftoprint['variance']=df[inpvar].std()**2
                    df_gof_large=dftoprint.reset_index(drop=True)
                    #st.write(df_gof_large.head(1))
                    dfresultMVG=dfresultMVG.append(df_gof_large.head(1),ignore_index=True)
                    #dfresultMVG.append(dftoprint)
            #st.caption("Summary of the fit of columns selected with normal distribution")        
            #st.write(dfresultMVG[['column','mean','variance','KStest','Chi squared Test','SSE']])
            fit_mult=st.button("Generate code for Random Variate Generation.")
            fulllist=[]
            for i in range(len(df[dfcols[0]])):
                inlist=[]
                for corr in multivars_inp:
                    #st.write(corr)
                    inlist.append(df[corr][i])
                    
                fulllist.append(inlist)
            if fit_mult: 
                #st.write("Please wait")
                
                gof_col,cov_col=st.columns(2)
                with gof_col:
                    #st.caption("Summary of the fit of columns selected with normal distribution")   
                    st.markdown('<p class="small-font">Summary of the fit of columns selected', unsafe_allow_html=True)
                    st.write(dfresultMVG[['column','mean','variance','KStest','Chi squared Test','SSE']])
                with cov_col:
                    cov=np.cov(fulllist, rowvar=False)
                    #st.write(fulllist)
                    st.markdown('<p class="small-font">Covariance matrix of the selected columns', unsafe_allow_html=True)
                    #st.caption("Covariance matrix of the selected columns")
                    st.write(cov)
                    
                    
                    
                    
                    
                    
                    
                st.markdown('<p class="small-font">Code to generate the multivariate Gaussian distribution. Adjust the num_datapoints parameter to change the number of points to generate', unsafe_allow_html=True)
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
                #st.markdown('<p class="small-font">', unsafe_allow_html=True)
    
                st.code(output)

       
    
        
        
        
                    
                
       
            
        
        

       
            
            
if __name__=='__main__':
    main()
        
    