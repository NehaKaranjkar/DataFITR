#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 02:34:24 2023

@author: lekshmi"""


import matplotlib.pyplot as plt
from scipy import stats
import pickle


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import sys,io,os
import streamlit as st
import pandas as pd
import scipy.stats as stats,time
from io import StringIO
#import plotly.graph_objects as go
#sys.path.append('/home/lekshmi/Downloads/my_app/IM') 
sys.path.append('./IM') 
#import modular_IM
from IM import modular_IM
from IM import kde_silverman_both
from IM import datagenforDataFITR
import datagenforDataFITR


#import kde_silverman_both
from IM import JDF_bivariate
import math
import re
from IM import listparamsofdistributions
import numpy as np
#from statsmodels.tsastattools import adfuller
import seaborn as sns
from scipy.stats import pearsonr #to calculate correlation coefficient
#import plotly.express as px
def to_csv(df):
    data_csv = df.to_csv(path_or_buf=None, sep=',', index=False) 
    return data_csv

def cdf(random_variable):
  x, counts = np.unique(random_variable, return_counts=True)
  cusum = np.cumsum(counts)
  cdf = cusum / cusum[-1]
  return x, cdf

def frequency_table(data, bins=50):
    #ax=np.histogram(data,bins)
    ax = plt.hist(data, bins=bins)
    plt.close()
    freqs = ax[0]
    intervals = ax[1]

    freq_table = {}

    for i in range(0, len(intervals)-1):
        freq_table[tuple([intervals[i], intervals[i+1]])] = int(freqs[i])

    return freq_table

def main():
    st.set_page_config(page_title="DataFITR",page_icon="📈",layout="wide")
    
    #st.sidebar.subheader("Fitting Arbitrary Multivariate Data")
    
    #st.sidebar.image("/home/lekshmi/Downloads/DataFitter/MISC/datafitrlogo.PNG", use_column_width=True)
    st.sidebar.image("./MISC/datafitrlogo.PNG", use_column_width=True)
    st.subheader("Fitting Multivariate Data with Arbitrary joint distributions")
    st.markdown("---")
    
    data_col,button_col=st.columns(2)
    with data_col:
        
        datainput_formMVA=st.radio("Select a data source",
                ('Upload a CSV','Use a sample synthetic data',))
        if datainput_formMVA=='Use a sample synthetic data':
            numdatagen=st.slider("Enter the number of datapoints to generate", 50, 5000,500)
       
        else:
            st.write("The expected format is as shown below. ")
            #st.image("./MISC/sampledata.PNG",width=500)
            st.markdown("""
            The tool expects the data to be in a csv format where the column name is the name of the process variable and the rows are the observations.
            """)
    with button_col:
        if datainput_formMVA=='Upload a CSV':
            try:
                
                st.markdown('''Please upload a CSV file with the variables to fit and proceed.''')
                uploaded_file = st.file_uploader("Choose a file.",type="csv")
                #st.session_state.data_generatedMAG=False
                if uploaded_file is None:
                    st.warning("Please upload a csv file to proceed")
                    return
                    
                        
                else:
                    dfMVA = pd.read_csv(uploaded_file)
                    st.session_state.data_generatedMVA=True
                    st.session_state.dataMVA=dfMVA
                    startcheck=dfMVA.describe(include="all")
                    
            except:
                st.warning("Please upload a csv file which adheres to the format mentioned in the documentation.")
                #uploaded_file = st.file_uploader("Choose a file.")
                #IM_uni(dfMVA)    
                
        
            
            #IM_uni(dfMVA)
    
        else:
            
            #st.session_state.data_generatedMVA=False
            if st.button(("Regenerate a sample synthetic data" if "data_generatedMVA" in st.session_state else "Generate a sample synthetic data")):
                
                dfMVA=datagenforDataFITR.arbitrary_MV_samples(numdatagen)
                st.session_state.data_generatedMVA=True
                st.session_state.dataMVA=dfMVA
        
                with st.expander("View raw data"):
                    st.write(dfMVA)
                st.download_button('Download generated data as a CSV file', to_csv(dfMVA), 'sample_multivariate_gaussian_data.csv', 'text/csv')
                
    if  'data_generatedMVA' in st.session_state:
        #st.write(st.session_state.data_generated)
        
        if st.button("Start Exploratory Data Analysis"):
            st.session_state.button_clickedMVA=True
    if 'button_clickedMVA' in st.session_state:
            datatofitMVA=st.session_state['dataMVA']
            #st.write(datatofit)    
            IM_MVA(datatofitMVA)
            
    
    
    
    
    
    
    
    
def IM_MVA(df):   
    
    st.header("Data Exploration")
      
    #df = pd.read_csv(uploaded_file)
    if len(df.columns)==1:
        numcolerror()
    else:
        #filenamevalue=uploaded_file.name
        #folder_name=filenamevalue.split()[0][:-4]#removing .csv from name
        folder_name='SampleMVA_CSVfile'
        fig= plt.figure(figsize=(15, 15))
        j=1
        cols=list(df.columns)
        
        st.subheader("Marginal distribution of the columns in the dataset")
        for i in cols:
            plt.subplot(len(cols),3,j)
            plt.tight_layout()
            a=sns.histplot(df[i],bins=100,kde=True)
            #a=sns.distplot(df[i],bins=100,rug=True)
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
        
    #plot to visualise joint density distribution
    st.write("Select the components to visualize data")
    col1,col2=st.columns(2)
    jdf_collist=list(df.columns)
    with col1:
        selected_x_axis=st.selectbox("x-axis", jdf_collist)
    with col2:
        selected_y_axis=st.selectbox("y-axis", [a for a in jdf_collist if (a!=selected_x_axis)])
    
    
        
        
   
    
    plot_contr,plot_jdf=st.columns(2)
    with plot_jdf:
        num_bins_jdf=st.slider("Select the number of bins to create grids for joint distribution",1,200,50,key='binstojdf')
        JDFans=JDF_bivariate.to_JDF(df[[selected_x_axis,selected_y_axis]],num_bins_jdf)
        copula=JDFans[0]
        dic_bins=JDFans[1]
        x=[]
        y=[]
        cols=list(dic_bins.keys())
        for i in copula.keys():
            lower=i[0]-1
            upper=i[1]-1
            x.append(dic_bins[cols[0]][lower])
            y.append(dic_bins[cols[1]][upper])
            
        
                
        z=[i/sum(copula.values()) for i in copula.values()]
        fig = plt.figure(figsize = (10, 10))
        ax = plt.axes(projection ="3d")
        #ax=Axes3D(fig)
        #ax.scatter3D(x,y, z, color="red")
         #
        # Creating plot
        #fig = go.Figure([go.Surface(x=x, y=y, z=z)])
        ax.plot_trisurf(x, y, z,linewidth = 0.2,antialiased = True,alpha=0.4 );
        ax.set_xlabel(selected_x_axis, fontweight ='bold')
        ax.set_ylabel(selected_y_axis, fontweight ='bold')
        ax.set_zlabel("Probability", fontweight ='bold')
        #plt.title("3D scatter plot")
        st.pyplot(fig)
        #st.plotly_chart(fig)
    with plot_contr:
        st.caption("Scatter plot of the raw values of the selected columns")
        fig1=plt.figure(figsize=(10,10))
        a=sns.jointplot(x=selected_x_axis,y=selected_y_axis,data=df)
        a.plot_marginals(sns.rugplot, height=-.15, color='black', clip_on=False)
        #a.plot_joint(sns.kdeplot, color='black', levels=7)
        #a.set_xlabel(selected_x_axis,fontsize=15)
        #a.set_ylabel(selected_x_axis,fontsize=15)
        st.pyplot(a)
        
        
    multivarsA_inp=st.multiselect("Select the variables with correlation to generate code for RVG", df.columns,key='multivarAcols')
    #st.write(list(multivarsA_inp),df)
    
    
    if multivarsA_inp:
        
        df1=df[multivarsA_inp]
        df_cdfvals = df1.copy(deep=True)
        df1=df_cdfvals.copy()
        for i in df_cdfvals.columns:
            col = df_cdfvals[i]
            x_sort_i, F_i = cdf(col)
            #print(matrix_F)
            F_cdf=[]
            for val in col:
                cdfval=np.where(x_sort_i==val)[0][0]
                F_cdf.append(F_i[cdfval])
            df_cdfvals[i]=F_cdf
        print(df_cdfvals)

        cdf_data_dict = {}
        num_bins=st.slider("select bin size",5,1000,50,key='groupbins')
        for i in df1.columns:
            count,binedges=np.histogram(df1[i],num_bins)
            pmf_col_dict={}
            for k in range(len(binedges)-1):
                binlim=tuple([binedges[k],binedges[k+1]])
                pmf_col_dict[binlim]=count[k]
            cumpmf=np.cumsum(list(pmf_col_dict.values()))
            cdfpmf=cumpmf/cumpmf[-1]
            
            cdf_col_dict={}
            ival=0
            for k in pmf_col_dict:
                cdf_col_dict[k]=cdfpmf[ival]
                ival+=1
              
            cdf_data_dict[i] = cdf_col_dict
        
        
        
        
        output= f""" 
        
       
       import pandas as pd
       import pickle
       import numpy as np
        
        
       with open('cdf_MVA.pkl','rb') as fp:
           cdf_data_dict=pickle.load(fp)
       with open('matrix.pkl','rb') as fp:
           df_cdfvals=pickle.load(fp)
       df_columns=list(cdf_data_dict.keys())
       data_gen = pd.DataFrame(columns=df_columns)
       rand_ind = np.random.randint(low=0, high=len(df_cdfvals), size=300)   
       for n in rand_ind:
           synth_gen = []
           for i in df_columns:
               h = df_cdfvals[i][n]
               binintr=-1
               for l in cdf_data_dict[i]:
                   #print(h, cdf_data_dict[i][l],i,l)
                   if h<=cdf_data_dict[i][l]:
                       binintr=l
                       break
               synth_gen.append(np.random.uniform(low=binintr[0], high=binintr[1], size=1)[0])

           synth_gen = np.array(synth_gen).T
           synth_gen = pd.DataFrame([synth_gen], columns=df_columns)
           data_gen = pd.concat([data_gen, synth_gen], ignore_index=True)
            """
        code_MVA,download_MVA=st.columns(2)
        with code_MVA:
            st.code(output)
        with download_MVA:
            st.caption("Use the pickle file along with the code to generate data")
            st.download_button("Click to download the cdf_MVA.pkl file", data=pickle.dumps(cdf_data_dict),file_name="cdf_MVA.pkl")
            st.download_button("Click to download the matrix.pkl file", data=pickle.dumps(df_cdfvals),file_name="matrix.pkl")
            picklfilename="MVA"+'pickle'
            st.session_state[picklfilename]=pickle.dumps(cdf_data_dict)
    
    
    
            
            
if __name__=='__main__':
    main()
        