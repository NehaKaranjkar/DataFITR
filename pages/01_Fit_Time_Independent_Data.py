#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:14:15 2023

@author: lekshmi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from scipy import stats

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
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
from IM import datagenforDataFITR
#import kde_silverman_both
#fromm IM import wrapper
import math
import re
from IM import listparamsofdistributions
import numpy as np
#from statsmodels.tsastattools import adfuller
import seaborn as sns
from scipy.stats import pearsonr #to calculate correlation coefficient



@st.cache(suppress_st_warning=(True))
def my_function(data,distlist,distributions,typ='continuous',dist='my1 distribution',bins=100,gof="ks",kde_bin=50):
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
    modelGUI=modular_IM.modelmatch(data ,typ,dist,bins,gof,distlist,distributions,kde_bin)
    #st.write(modelGUI.data)
    plotdata,pval=modelGUI.bestfit(distlist,distributions)
    result,SSEresult=modelGUI.printresult()
    #result.rename({'key_0': 'Test'}, axis=1, inplace=True)
    
    #st.write(pd.DataFrame(plotdata))
    restyp='nonconstant'
    
    
    
    #st.write(plotdata)
    #st.write("checkode")
    #st.write(list(result.index))
    #st.write(result)
    return restyp,result,plotdata,pval

def dfinfo(df):
    buffer=io.StringIO()
    df.info(buf=buffer)
    lines = buffer.getvalue ().split ('\n')
    #st.write(lines)
    # lines to print directly
    #lines_to_print = [0, 1, 2, -2, -3]
    #for i in range(len(lines)):
        #st.write (lines [i])
    # lines to arrange in a df
    list_of_list = []
    for x in lines [5:-3]:
        listouter = x.split ()
        listinner=[listouter[0],]
        shrt=""
        for i in range(1,len(listouter)-3):
            shrt+=listouter[i]
        #st.write(shrt,listouter[-3:])
        listinner.append(shrt)
        listinner.extend(listouter[-3:])
       
            
        
            
            
            
        list_of_list.append (listinner)
        
    info_df = pd.DataFrame (list_of_list, columns=['index', 'Column', 'Non-null-Count', 'null', 'Dtype'])
    info_df.drop (columns=['index', 'null'], axis=1, inplace=True)
    #st.dataframe(info_df)
    
    return info_df
    

@st.cache(suppress_st_warning=(True))
def getbinsize():
    bin_size=st.text_input("Enter an approriate bin size for the data to plot histogram")
    if bin_size:
        return bin_size
    else:
        st.write("Enter a binsize to proceed")
        
@st.cache(suppress_st_warning=(True))
def getcontinuousdist():
    Continuous_All=[]
    all_dist = [getattr(stats, d) for d in dir(stats) if isinstance(getattr(stats, d), stats.rv_continuous)]
    filtered = [x for x in all_dist if ((x.a <= 0) & (x.b == math.inf))]
    filtered=all_dist
    pat = r's.[a-zA-Z0-9_]+_g'
    
    for i in filtered:
        s = str(i)
        #print(s)
        span=re.search(pat, s).span()
        dist=s[span[0]+2:span[1]-2]
        Continuous_All.append(dist)
        
    for i in ['levy_stable','studentized_range','kstwo','skew_norm','vonmises','trapezoid','reciprocal']:
        if i in Continuous_All: 
            Continuous_All.remove(i)
    return Continuous_All
@st.cache
def getdiscretedist():
    discrete=['binom','poisson','geom']
    return discrete

def ifpathExists(path):
    isExist=os.path.exists(path)
    if isExist:
        return True
    else:
        return False
    
def ifFileExists(path,filename):
    dir_list=os.listdir(path)
    if filename in dir_list:
        return True
    else:
        return False
    
def fileCreate(path,filename):
    os.chdir(path)
    #df=pd.DataFrame(list())
    df=pd.DataFrame()
    df.to_csv(filename)
    
def updateFile(path,df):
    df.to_csv(path,mode='w+',index=False)
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    #os.remove(path)
    #df.to_csv(path,mode='a',index=False)

def pearsonCorr(x, y, **kws): 
    (r, _) = pearsonr(x, y) #returns Pearson’s correlation coefficient, 2-tailed p-value)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),xy=(.7, .9), xycoords=ax.transAxes)



def is_CSV(filename):
   #st.write(filename.split("."))
   return filename.split(".")[1]

def to_csv(df):
    data_csv = df.to_csv(path_or_buf=None, sep=',', index=False) 
    return data_csv

        
def checkdatatype(df):
    for i in df.columns:
        if (df[i]).dtype  not in[ 'float','int']:
            return 0
    return 1






@st.experimental_memo
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

def finddatatype_autofill(data):
    #g=[True if i**2== (int(i))**2 else False  for i in data]
    g=[True if (float(i)-abs(i))==0 else False  for i in data]
    if data.dtype=='int64':
        return 'Integer-Valued'
    elif False in g:
        return 'Real-Valued'
    else:
        return 'Real-Valued'
    
    
    
    
    
def IM_uni(df):
   
        
    st.header("Data Exploration")
    if checkdatatype(df)==0:
 
        st.warning("All categorical columns should be converted to numerical values")
        return
    
    with st.expander("Click here to view the results of Exploratory Data Analysis"): 
        #folder_name='SampleCSVfile'
        #path='./output'     
        #foldername=st.text_input("Enter a name for the folder to store the results",value=folder_name,key='foldername')                   
        st.sidebar.header("Dataset Summary")
        dataset_shape=('The dataset has {} variables and {} observations.').format(df.shape[1],df.shape[0])
        st.sidebar.write(dataset_shape)        
        dtype_col,desc_col=st.columns(2)   
        
        with desc_col:
            #st.caption("Summary of the dataset uploaded")
            st.markdown("""<style>.big-font {font-size:20px !important;}</style>""", unsafe_allow_html=True)
            st.markdown('<p class="big-font">Summary of the dataset uploaded', unsafe_allow_html=True)
            
            st.write(df.describe(include="all"))     
            
        with dtype_col:
            #st.caption("Data type of the columns in the dataset")
            st.markdown("""   """)
            st.markdown('<p class="big-font">Data ype of the columns in the dataset', unsafe_allow_html=True)           
            st.dataframe(dfinfo(df))
        if len(list(df.columns))>1:
            fig= plt.figure(figsize=(15, 15))
            j=1
            cols=list(df.columns)
            st.subheader("Marginal distribution of the columns in the dataset")
            for i in cols:
                plt.subplot(len(cols),3,j)
                plt.tight_layout()
                #dflist=np.array(df[i])
                a=sns.histplot(df[i],bins=100,kde=True)
                #mean=round(df[i].mean(),2)
                #std=round(df[i].std(),2)
                #texttoprint= "mean="+str(mean)+'\n'+"stddev="+str(std)
                #a.axvline(df[i].mean(),color='k',lw=2)
                
                a.set_xlabel(i,fontsize=15)
                a.set_ylabel("density",fontsize=15)
                j=j+1
            st.pyplot(fig)
            cols=list(df.columns)
        
        
        
    st.header("Detecting Correlation")
        
        
        
    with st.expander("Click here to view correlation between columns",): 
            
            
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
            rotx = a.set_xticklabels(a.get_xticklabels(), rotation=25,size=15)
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
        
        
        if len(list(df.columns))==1:
            #fig= plt.figure(figsize=(5,5))
            cols=list(df.columns)
            st.subheader("Marginal distribution of the columns in the dataset")
            for i in cols:
                fig,ax=plt.subplots(1,1)
                plt.tight_layout()
                a=sns.histplot(df,x=i,bins=100,kde=True)
                a.set_xlabel(i,fontsize=15)
                a.set_ylabel("density",fontsize=15)
            st.pyplot(fig)
            st.header("Detecting Correlation")
            st.info("There is only one data column in the uploaded file. Not enough features to find the correlation.")

    
    

    
   
    
    
    st.header("Input Modeling for non correlated and time independent data")
    #st.caption("Use this if the data is non correlated and time independent")
    #st.markdown('<p class="big-font">Fit non correlated and time independent data', unsafe_allow_html=True)
        
    datadetailscol,histcol=st.columns(2)
    
    with datadetailscol:
        #check#dataname = st.text_input("Enter a name for the dataset",value="sample.csv",key='dataname')  
        st.subheader("Fitting data columnwise")
        st.warning("Choose a data column from the CSV file to fit/model")
        inpvar = st.selectbox("data column", df.columns)
        
        
        st.sidebar.write("Attribute chosen by the user to perform input modeling is",inpvar)
        datatype_autofilloption=finddatatype_autofill(df[inpvar])
        datatype_list=['Real-Valued','Integer-Valued']
        default_ind=datatype_list.index(datatype_autofilloption)
        #st.write(default_ind)
        
        if datatype_list[default_ind]=='Real-Valued':
            datatype_option = st.selectbox('Datatype of the column:', ['Real-Valued','Integer-Valued'],index=default_ind)
        else:
            datatype_option = st.selectbox('Datatype of the column:', datatype_list,index=default_ind)
            st.warning("The datatype of the column is inferred from the data. Click on the dropdown if you wish to change it.")
        Continuous_All=getcontinuousdist()[70:80]
        Continuous_Popular=['expon','norm','lognorm','triang','uniform','weibull_min','gamma']
        Discrete_Popular=['binom','poisson','geom']
        if inpvar:

            
            #st.sidebar.write("Dataset name:",dataname)
            datatyp='na'
            if datatype_option=='Real-Valued':
                datatyp='continuous'
                distlist_userselect = st.selectbox("Choose distributions to fit", ['Continuous_Popular','Continuous_All',],key='dislist_user')
            else:
                datatyp='discrete'
                distlist_userselect = st.selectbox("Choose distributions to fit", ['Discrete_Popular',],key='dislist_user')
        
        
    
            
        
        
    with histcol:
        st.subheader("Histogram of the selected column")
        st.warning("Move the slider to change the binsize")
        datalen=len(df[inpvar])
        defaultbin=min((1+np.ceil(np.log(datalen))),max(100,datalen/10))
        maxbin=max(200,10*defaultbin)
        
        bins=st.slider("Choose an appropriate number of bins to plot the histogram.",1,int(maxbin),int(defaultbin),key='bins')
        st.sidebar.write("The binsize is ",bins)
        fig,ax=plt.subplots()
        ax.hist(df[inpvar],edgecolor = "black",bins=bins,density = 1)
        plt.xlabel('data values')
        plt.ylabel('pdf')
        plt.title('Histogram of data stream selected')
        st.pyplot(fig)
        
        
    
    
    
    distlist=[]
    distributions=[]
    if (('Continuous_All' in distlist_userselect) or ('Continuous_Popular'  in distlist_userselect)):
        distlist.append('continuous')
        st.session_state['distlist']='continuous'
        if 'Continuous_All' in distlist_userselect:
            Continuous_Alldist=getcontinuousdist()
            distributions.append(Continuous_Alldist)
        else:
            distributions.append(Continuous_Popular)
    if (('Discrete_All' in distlist_userselect) or ('Discrete_Popular'  in distlist_userselect)):
        distlist.append('discrete')
        st.session_state['distlist']='discrete'
        if 'Discrete_All' in distlist_userselect:
            Discrete_Alldist=getdiscretedist()
            distributions.append(Discrete_Alldist)
        else:
            distributions.append(Discrete_Popular)
        
    if st.button("Start fitting the data"):
        st.session_state.fit_button=True
    
    if 'fit_button' in st.session_state:
        #st.write(df[inpvar],distlist,distributions,datatyp,inpvar,bins,'ks',50)
        #st.write(my_function(df[inpvar],distlist,distributions,datatyp,inpvar,bins,'ks',50))
        restyp,finresult,plotdata,pval=my_function(df[inpvar],distlist,distributions,datatyp,inpvar,bins,'ks')
        if restyp=='constant':
            
            constdict={"column":inpvar,"value":[pval],'type':['constant']}
            constdf=pd.DataFrame(constdict)
            st.write(constdf)
            #updateFile(fullpath,constdf)
            constantname=inpvar+'constant'
            st.session_state[constantname]=constdf
            st.info("The data to fit is a contant value and the value is  "+str(pval))
            
        else:
            #st.write(finresult)
            cols=list(finresult.columns)[1:]
            goodnessoffit=st.selectbox("Select a goodness of fit measure",cols,key='goodnessoffit')
            
            
           
            
            #GOF=pd.DataFrame(GOF)
            #SSE=pd.DataFrame(SSE)
            #g=GOF.transpose()
            #g=g.astype(str)
            #st.write(g)
            if 'continuous' in distlist:
                #contcol,GOFcol,kdecol=st.columns(3)
                contcol,kdecol=st.columns(2)
                with contcol:
                    st.markdown("""<style>.big-font {font-size:18px !important;}</style>""", unsafe_allow_html=True)
                    st.markdown('<p class="medium-font">The histogram of the data and the line plots of various matched distribution.', unsafe_allow_html=True)
                    st.markdown("""   """)
                    st.markdown("""   """)
                    st.markdown("""   """)
                    st.markdown("""   """)
                    if goodnessoffit:
                        #st.write(plotdata)
                        
                        
                        result_df = finresult.sort_values(by = goodnessoffit)
                        #colstoplot=list(result_df.index[:5])
                        colstoplot=list(result_df.head(5)['Test'])
                        if 'kde' in colstoplot:
                            colstoplot.remove("kde")
                        colors = ['r', 'g', 'm', 'y', 'k', 'w', 'orange', 'purple','c']
                        fig, ax = plt.subplots()
    
                        # Plot the histogram
                        
                        df_plot = pd.DataFrame(plotdata)
                        #st.write(bins)
                        #ax.hist(df[inpvar],edgecolor = "black",bins=bins,density = 1)
                        ax.hist(df_plot['data'], bins,label='Histogram', density=True, color='b', alpha=0.4)
                        
                        # Plot the line plots
                        j=0
                        for i in colstoplot:
                            #st.write(i,df_plot[i])
                            ax.plot(df_plot['data'],df_plot[i], label=i,color=colors[j])
                            j+=1
                        ax.legend(fontsize=15)
                        ax.set_title('Histogram and Line Plots',size=18)
                        ax.set_xlabel('x -values',size=16)
                        ax.set_ylabel('pdf',size=16)
    
                        # Show the plot in Streamlit
                        st.pyplot(fig)
                        imagename=inpvar+'image'
                        st.session_state[imagename]=fig
                        #fig.savefig(fullpathfig)#filesave
    
                        
                
                        
                        
                with kdecol:
                    if goodnessoffit:
                        #objplt.histogram(df_plot.index,bins=bins,density = 1,alpha=0.4)
                        #st.write(objplt)
                        #st.markdown("""<style>.big-font {font-size:18px !important;}</style>""", unsafe_allow_html=True)
                        #st.markdown('<p class="big-font">The histogram of the data and Kernel Density Estimation plot', unsafe_allow_html=True)
                        #st.markdown('<p class="big-font">The histogram of the data and Kernel Density Estimation plot', unsafe_allow_html=True)
                        st.markdown("""   """)
                        kdeh=st.slider("Choose an appropriate smoothing parameter, h for KDE plot",5,200,2*bins,key="kdeh")
                        kdecentre,kdepdf,h=kde_silverman_both.kde_func(df[inpvar], inpvar, kdeh)
                        restyp,finresult,plotdata,pval=my_function(df[inpvar],distlist,distributions,datatyp,inpvar,bins,'ks',kdeh)
                        #dfkde=pd.DataFrame({'kdecentre':kdecentre,"kdepdf":kdepdf})
                        #dfkde=dfkde.set_index('kdecentre')
                        fig,ax=plt.subplots()
                        ax.hist(df[inpvar],edgecolor = "black",label='Histogram',bins=bins,density = 1)
                        ax.plot(kdecentre,kdepdf, label="kde",color="red")
                        ax.legend(fontsize=15)
                        plt.xlabel('data values',size=16)
                        plt.ylabel('pdf',size=16)
                        plt.title('Histogram of data stream selected',size=18)
                        st.pyplot(fig)
                        
                        #st.line_chart(dfkde)
                st.markdown('<p class="big-font">Goodness of fit values(Lower is better)', unsafe_allow_html=True)
                if goodnessoffit:
                    result_df = finresult.sort_values(by = goodnessoffit)
                    up_df=result_df.copy()
                    #up_df['var']=filename
                    #st.write(fullpath)
                    #updateFile(fullpath,up_df)#filesave
                    st.session_state['df']=up_df
                    #st.write(up_df)
                    dftoprint=result_df.head(10)
                    df_gof_large=dftoprint.reset_index(drop=True)
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
                    df2=df_gof_large.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles)
                    st.table(df2)
                    tabname=inpvar+'table'
                    st.session_state[tabname]=df2
                    #st.write(list(up_df['Test']))
                    #st.write(result_df.head(5))
                    
                    #st.table(SSE)
            else:
                
                disccol,GOFcol=st.columns(2)
                with disccol:
                    
                    if goodnessoffit:
                        #st.write(goodnessoffit)
                        #st.write(plotdata)
                        
                        
                        result_df = finresult.sort_values(by = goodnessoffit)
                        #colstoplot=list(result_df.index[:5])
                        colstoplot=list(result_df.head(5)['Test'])
                        if 'kde' in colstoplot:
                            colstoplot.remove("kde")
                        colors = ['r', 'g', 'm', 'y', 'k', 'w', 'orange', 'purple','c']
                        fig, ax = plt.subplots()
    
                        # Plot the histogram
                        
                        
                        bindetails=np.histogram(df[inpvar],bins=bins)
                        bincount=bindetails[0]
                        binpdf=bincount/bincount.sum()
                        binedges=bindetails[1]
                        binedgelast=binedges[-1]
                        np.append(binedges,binedgelast)
                        #binedges=np.round(bindetails[1])
                        #st.write(bindetails,binpdf,binedges)
                        ax.hist(df[inpvar],bins=bins,density = True,color='b',alpha=0.4)
                        #ax.bar(binedges,binpdf,width=1)
                        #ax.hist(df[inpvar],edgecolor = "black",bins=bins,density = 1)
                        
                        
                        df_plot = pd.DataFrame(plotdata)
                        
                   
                        # Plot the line plots
                        j=0
                        for i in colstoplot:
                            #st.write(i,df_plot[i])
                            ax.plot(df_plot['data'],df_plot[i], label=i,color=colors[j])
                            j+=1
                        ax.legend(fontsize=15)
                        ax.set_title('Histogram and Line Plots',size=18)
                        ax.set_xlabel('x -values',size=16)
                        ax.set_ylabel('pdf',size=16)
    
                        # Show the plot in Streamlit
                        st.pyplot(fig)
                        #fig.savefig(fullpathfig)#filesave
                        imagename=inpvar+'image'
                        st.session_state[imagename]=fig
                        
                        
                        
    
                        #df_plot = pd.DataFrame(plotdata)
                        #df_plot=df_plot.set_index('data')
                        #objplt=st.line_chart(df_plot)
                        
                        
                with GOFcol:
                    if goodnessoffit:
                        st.subheader("Goodness of fit test results")
                        result_df = finresult.sort_values(by = goodnessoffit)
                        up_df=result_df.copy() 
                        st.session_state['df']=up_df
                        #up_df['var']=filename
                        #updateFile(fullpath,up_df)#filesave
                        
                        #st.write("update")
                        #st.write(list(up_df['Test']))
                        dftoprint=result_df.head(10)
                        st.write(dftoprint.reset_index(drop=True))
                        tabname=inpvar+'table'
                        st.session_state[tabname]=dftoprint
                
                        #st.write(result_df.head(5))
            with st.expander("Zoom and show the line plots"):
                selectcol,plotcol=st.columns(2)
                with selectcol:
                    disttozoom=st.selectbox("choose a distribution to plot", colstoplot)
                with plotcol:
                    fig, ax = plt.subplots()
                   
                    ax.plot(df_plot['data'],df_plot[disttozoom], label=disttozoom,color=colors[j])
                    
                    ax.legend(fontsize=15)
                    ax.set_title('Line Plots',size=18)
                    ax.set_xlabel('x -values',size=16)
                    ax.set_ylabel('pdf',size=16)

                    # Show the plot in Streamlit
                    st.pyplot(fig)
                        #st.table(SSE)
            distlist=list(up_df['Test'])
            #distlist=list(up_df.index)
            if distlist[0]!='kde':
                params=listparamsofdistributions.getparams(distlist[0],datatyp,df_plot['data'])
                string_info='The data column "{}" selected to fit has matched with "{}" with parameters {}.'.format(inpvar,distlist[0],params)   
                st.info(string_info)  
            else:
                string_info='The data column "{}" selected to fit has matched with "{}" .'.format(inpvar,distlist[0])   
                st.info(string_info) 
                

            st.header("Random Variate Generation")               
            with st.expander("Click here to get the code for Random Variate Generation"):               
                #distlist=list(up_df['Test']) 
                #params=listparamsofdistributions.getparams(distlist[0],datatyp,df_plot['data'])
                #string_info='The data column "{}" selected to fit has matched with "{}" with parameters {}'.format(inpvar,distlist[0],params)   
                #st.info(string_info)   
                #distlist=list(plotdata.keys())[1:]
                
                st.text("Code for random variate generation for the column "+ str(inpvar))
                #st.write(list(plotdata.s())[0])
                dist_userselect=st.selectbox("Select the distribution of your choice to generate code. Adjust the num_datapoints parameter to change the number of points to generate",distlist )
                #st.write(st.session_state)
                if dist_userselect:
                    if dist_userselect=='kde':
                        code,download=st.columns(2)
                        with code:
                            output=listparamsofdistributions.gencodekde(df_plot['data'],inpvar)
                            st.code(output)
                            code=inpvar+"code"
                            st.session_state[code]=output
                        with download:
                            st.caption("Use the pickle file along with the code to generate data")
                            kernel=stats.gaussian_kde(df_plot['data'])
                            st.download_button("Click to download the Kernel.pkl file", data=pickle.dumps(kernel),file_name="kernel.pkl")
                            picklfilename=inpvar+'pickle'
                            st.session_state[picklfilename]=pickle.dumps(kernel)
                        
                        
                        
                        
                        
                    else:
                        output=listparamsofdistributions.gencode(dist_userselect, datatyp, df_plot['data'])
                        st.code(output)
                        code=inpvar+"code"
                        st.session_state[code]=output
                        picklfilename=inpvar+'pickle'
                        if picklfilename in st.session_state:
                            del st.session_state[picklfilename]
                    
                
                        
        
                
               
            
           
                    
            
            






def main():
    
    st.set_page_config(page_title="DataFITR",page_icon="📈",layout="wide")



    st.subheader("Fitting Time Independent Data")
    st.markdown("---")


    st.sidebar.markdown('''
    # Sections
    - [Data Exploration](#data-exploration)
    - [Detecting correlation](#detecting-correlation)
    - [Input Modeling for non correlated & time independent data](#input-modeling-for-non-correlated-&-time-independent-data)
    ''', unsafe_allow_html=True)

    st.sidebar.image("./MISC/datafitrlogo.PNG", use_column_width=True)
    
    st.sidebar.subheader("Fitting Time Independent Data")
    
    
    data_col,button_col=st.columns(2)
    with data_col:
        
        datainput_form=st.radio("Select a data source",
                ('Upload a CSV','Use a sample synthetic data','Use an open data set'))
        if datainput_form=='Use a sample synthetic data':
            numdatagen=st.slider("Enter the number of datapoints to generate", 50, 5000,2000)
        elif datainput_form=="Use an open data set":
            st.write("Generating datapoints from open dataset")
        else:
            st.write("The expected format is as shown below. ")
            st.image("./MISC/sampledata.PNG",width=500)
            st.markdown("""
            The tool expects the data to be in a csv format where the column name is the name of the process variable and the rows are the observations. Categorical data should be converted into numerical values.
            """)
    
        
            
        
    with button_col:
        if datainput_form=='Upload a CSV':
            try:
                st.markdown('''Please upload a CSV file with the variables to fit and proceed.''')
                uploaded_file = st.file_uploader("Choose a file.",type="csv")
                #st.session_state.data_generated=False
                if uploaded_file is None:
                    st.warning("Please upload a csv file to proceed")
                    return
                    
                        
                else:
                    df1 = pd.read_csv(uploaded_file)
                    st.session_state.data_generated=True
                    st.session_state.data=df1
                    startcheck=df1.describe(include="all")
                    
            except:
                st.warning("Please upload a csv file which adheres to the format mentioned in the documentation.")
                #uploaded_file = st.file_uploader("Choose a file.")
                #IM_uni(df1)    
                
        
            
            #IM_uni(df1)
    
        elif datainput_form=='Use a sample synthetic data': 
            #st.session_state.data_generated=False
            if st.button(("Regenerate a sample synthetic data" if "data_generated" in st.session_state else "Generate a sample synthetic data")):
                
                df1=datagenforDataFITR.univariate_sample(numdatagen)
                st.session_state.data_generated=True
                st.session_state.data=df1
                #st.download_button('Download generated data as a CSV file', to_csv(df1), 'sample_data.csv', 'text/csv')
                with st.expander("View raw data"):
                    st.write(df1)
                st.download_button('Download generated data as a CSV file', to_csv(df1), 'sample_data.csv', 'text/csv')
            
        else:
            st.info("This dataset is an open data from Kaggle. A categorical column in the dataset is converted into numerical values")
            st.markdown("[Click here for details of the dataset](https://www.kaggle.com/datasets/gabrielsantello/parts-manufacturing-industry-dataset)")
            if st.button(("Reload open data" if "data_generated" in st.session_state else "Load open data")):
                
                df1=pd.read_csv("./MISC/opendata.csv")
                st.session_state.data_generated=True
                st.session_state.data=df1
                
                with st.expander("View open data"):
                    st.write(df1)
                st.download_button('Download open data as a CSV file', to_csv(df1), 'open_data.csv', 'text/csv')
            
     
            
    #st.session_state.button_clicked=False        
    #if 'data_generated' in st.session_state:    
        
            
            
       
    #st.write(st.session_state.data_generated)
    if  'data_generated' in st.session_state:
        #st.write(st.session_state.data_generated)
        
        if st.button("Start Exploratory Data Analysis"):
            st.session_state.button_clicked=True
    if 'button_clicked' in st.session_state:
            datatofit=st.session_state['data']
            IM_uni(datatofit)
            #st.write(datatofit)
            
    #st.write(st.session_state.data_generated)
    
            
            
            
            
                 
if __name__=='__main__':
    main()




    
    

        

        



 
    

    

            



    


