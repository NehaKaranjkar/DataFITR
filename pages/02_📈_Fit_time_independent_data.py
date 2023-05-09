#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:14:15 2023

@author: lekshmi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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



@st.cache(suppress_st_warning=(True))
def my_function(data,distlist,distributions,typ='continuous',dist='my1 distribution',bins=100,gof="ks",):
    st.write("Please wait while I fit your data")
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
    
@st.cache    
def gettypeofdata(dropdown):
    numitems=df.nunique()
    typ="na"
    bin_size=100
    dropdownlist=[]
    if numitems[dropdown]>(len(df)/4):
        
        st.write(dropdown,"is continuous")
        typ= "continuous"
        
    else:
        st.write(dropdown,"is discrete")
        typ= "discrete"
    return typ
@st.cache(suppress_st_warning=(True))
def getbinsize():
    bin_size=st.text_input("Enter an approriate bin size for the data to plot histogram")
    if bin_size:
        return bin_size
    else:
        st.write("Enter a binsize to proceed")
        
@st.cache(suppress_st_warning=(True))
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


st.set_page_config(page_title="DataFITR",page_icon="📈",layout="wide")



st.subheader("Modeling of time independent data")
st.markdown("---")


st.sidebar.markdown('''
# Sections
- [Data Exploration](#data-exploration)
- [Dectecting correlation](#detecting-correlation)
- [Input Modeling for non correlated & time independent data](#input-modeling-for-non-correlated-&-time-independent-data)
''', unsafe_allow_html=True)




@st.experimental_memo
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

def finddatatype_autofill(data):
    
    if data.dtype=='int64':
        return 'integer valued'
    else:
        return 'real valued'
    
    
    
    
    
def IM_uni(uploaded_file,uploaded_file_format,df):
    if uploaded_file_format  in['csv','CSV'] :
        
        st.header("Data Exploration")
          
        #df = pd.read_csv(uploaded_file)
        #st.write(df)
        filenamevalue=uploaded_file.name
        folder_name=filenamevalue.split()[0][:-4]#removing .csv from name
        folder_name=filenamevalue.split(".")[0]#removing .csv from name
        
        path='./output'
        #path='/home/ec2-user/DataFITR/output'
        
        foldername=st.text_input("Enter a name for the folder to store the results",value=folder_name,key='foldername')
        path=path+"/"+foldername
        #st.session_state['foldername']=foldername
        if ifpathExists(path)==False:
            os.makedirs(path)
            
        
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
            st.markdown('<p class="big-font">Data type of the columns in the dataset', unsafe_allow_html=True)
            
            
            st.dataframe(dfinfo(df))
        if len(list(df.columns))>1:
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
                rotx = a.set_xticklabels(a.get_xticklabels(), rotation=25,size=15)
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
            
            
        else:
            #fig= plt.figure(figsize=(5,5))
            cols=list(df.columns)
            st.subheader("Marginal distribution of the columns in the dataset")
            for i in cols:
                fig,ax=plt.subplots(1,1)
                plt.tight_layout()
                a=sns.distplot(df[i],bins=100,rug=True)
                a.set_xlabel(i,fontsize=15)
                a.set_ylabel("density",fontsize=15)
            st.pyplot(fig)
            st.header("Detecting Correlation")
            st.info("There is only one data column in the uploaded file. Not enough features to find the correlation.")

        
        

        
       
        
        
        st.header("Input Modeling for non correlated and time independent data")
        #st.caption("Use this if the data is non correlated and time independent")
        st.markdown('<p class="big-font">Fit non correlated and time independent data', unsafe_allow_html=True)
            
        datadetailscol,histcol=st.columns(2)
        
        with datadetailscol:
            dataname = st.text_input("Enter a name for the dataset",value="sample.csv",key='dataname')  
            
            inpvar = st.selectbox("Choose the data column to fit/model", df.columns)
            
            
            st.sidebar.write("Attribute chosen by the user to perform input modeling is",inpvar)
            datatype_autofilloption=finddatatype_autofill(df[inpvar])
            datatype_list=['real valued','integer valued']
            default_ind=datatype_list.index(datatype_autofilloption)
            #st.write(default_ind)
            
            if datatype_list[default_ind]=='real valued':
                datatype_option = st.selectbox('Datatype of the column:', ['real valued'],index=default_ind)
            else:
                datatype_option = st.selectbox('Datatype of the column:', datatype_list,index=default_ind)
                st.warning("The datatype of the column is inferred from the data. Click on the dropdown if you wish to change it.")
            continuous_all=getcontinuousdist()[70:80]
            continuous_popular=['expon','norm','lognorm','triang','uniform','weibull_min','gamma']
            discrete_popular=['binom','poisson','geom']
            if dataname:

                
                st.sidebar.write("Dataset name:",dataname)
                datatyp='na'
                if datatype_option=='real valued':
                    datatyp='continuous'
                    distlist_userselect = st.multiselect("Choose distributions to fit", ['continuous_all','continuous_popular'],default='continuous_popular',key='dislist_user')
                else:
                    datatyp='discrete'
                    distlist_userselect = st.multiselect("Choose distributions to fit", ['discrete_all','discrete_popular'],default='discrete_popular',key='dislist_user')
            
            old_fileclear=False
            fname=st.text_input("enter a name for the output file where the goodness of fit values will be stored",key='opfilename',value=inpvar)
            filename=fname+".csv"
            fullpath=path+"/"+filename
            fullpathfig=path+"/"+fname+'.png'
            
            if old_fileclear and ifFileExists(path, filename):
                os.chdir(path)
                os.remove(filename)
                fileCreate(path,filename)
            elif ifFileExists(path, filename)==False:
                fileCreate(path,filename)
            elif old_fileclear==0 and ifFileExists(path, filename)==1:
                os.chdir(path)
                with open(filename) as fp:
                    pass
                    #fp.write("#appending")
            else:
                pass
            os.chdir("../..")
        
                
            
            
        with histcol:
            st.subheader("Histogram of the selected column")
            st.warning("Move the slider to change the binsize")
            bins=st.slider("Choose an appropriate number of bins to plot the histogram.To fix the number of bins, you may  move the slider until the histogram looks like a continuous block with a distinct pattern",1,200,1,key='bins')
            st.sidebar.write("The binsize is ",bins)
            fig,ax=plt.subplots()
            ax.hist(df[inpvar],edgecolor = "black",bins=bins,density = 1)
            plt.xlabel('data values')
            plt.ylabel('pdf')
            plt.title('Histogram of data stream selected')
            st.pyplot(fig)
            distlist=[]
            distributions=[]
            if (('continuous_all' in distlist_userselect) or ('continuous_popular'  in distlist_userselect)):
                distlist.append('continuous')
                if 'continuous_all' in distlist_userselect:
                    continuous_alldist=getcontinuousdist()
                    distributions.append(continuous_alldist)
                else:
                    distributions.append(continuous_popular)
            if (('discrete_all' in distlist_userselect) or ('discrete_popular'  in distlist_userselect)):
                distlist.append('discrete')
                if 'discrete_all' in distlist_userselect:
                    discrete_alldist=getdiscretedist()
                    distributions.append(discrete_alldist)
                else:
                    distributions.append(discrete_popular)
            
                
        if st.checkbox("Start fitting the data"):
            #st.write(distlist,distributions,datatyp)
            restyp,finresult,plotdata,pval=my_function(df[inpvar],distlist,distributions,datatyp,dataname,bins,'ks',)
            if restyp=='constant':
                st.write("The data to fit is a contant value and the value is  "+str(pval))
                constdict={"value":[pval],'type':['constant'],'var':[inpvar+'.csv']}
                constdf=pd.DataFrame(constdict)
                st.write(constdf)
                updateFile(fullpath,constdf)
                
            else:
            
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
                        st.markdown('<p class="big-font">The histogram of the data and the line plots of various matched distribution.', unsafe_allow_html=True)
        
                        if goodnessoffit:
                            #st.write(plotdata)
                            
                            
                            result_df = finresult.sort_values(by = goodnessoffit)
                            
                            colstoplot=list(result_df.head(5)['Test'])
                            if 'kde' in colstoplot:
                                colstoplot.remove("kde")
                            colors = ['r', 'g', 'm', 'y', 'k', 'w', 'orange', 'purple','c']
                            fig, ax = plt.subplots()
        
                            # Plot the histogram
                            
                            df_plot = pd.DataFrame(plotdata)
                            ax.hist(df_plot['data'], bins=bins,label='Histogram', density=True, color='b', alpha=0.4)
        
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
                            fig.savefig(fullpathfig)
        
                            
                    
                            
                            
                    with kdecol:
                        if goodnessoffit:
                            #objplt.histogram(df_plot.index,bins=bins,density = 1,alpha=0.4)
                            #st.write(objplt)
                            kdeh=st.slider("Choose an appropriate smoothing parameter, h for KDE plot",5,200,5,key="kdeh")
                            kdecentre,kdepdf,h=kde_silverman_both.kde_func(df[inpvar], dataname, kdeh)
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
                    st.markdown('<p class="big-font">Table with goodness of fit values', unsafe_allow_html=True)
                    if goodnessoffit:
                        result_df = finresult.sort_values(by = goodnessoffit)
                        up_df=result_df.copy()
                        up_df['var']=filename
                        #st.write(fullpath)
                        updateFile(fullpath,up_df)
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
                        #st.write(result_df.head(5))
                        
                        #st.table(SSE)
                else:
                    
                    contcol,GOFcol=st.columns(2)
                    with contcol:
                        if goodnessoffit:
                            #st.write(goodnessoffit)
                            #st.write(plotdata)
                            
                            
                            result_df = finresult.sort_values(by = goodnessoffit)
                            
                            colstoplot=list(result_df.head(5)['Test'])
                            if 'kde' in colstoplot:
                                colstoplot.remove("kde")
                            colors = ['r', 'g', 'm', 'y', 'k', 'w', 'orange', 'purple','c']
                            fig, ax = plt.subplots()
        
                            # Plot the histogram
                            
                            #discrete    
                            binedges=pval['binedges']
                            binpdf=pval['binpdf']
                            ax.bar(binedges,binpdf,width=0.1)
                            
                            
                            
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
                            fig.savefig(fullpathfig)
                            
        
                            
        
                            #df_plot = pd.DataFrame(plotdata)
                            #df_plot=df_plot.set_index('data')
                            #objplt=st.line_chart(df_plot)
                            
                            
                    with GOFcol:
                        if goodnessoffit:
                            st.subheader("Goodness of fit test results")
                            result_df = finresult.sort_values(by = goodnessoffit)
                            up_df=result_df.copy() 
                            st.session_state['df']=up_df
                            up_df['var']=filename
                            updateFile(fullpath,up_df)
                            
                            #st.write("update")
                            #st.write(up_df)
                            dftoprint=result_df.head(10)
                            st.write(dftoprint.reset_index(drop=True))
                            #st.write(result_df.head(5))
                            
                            #st.table(SSE)
                            
                            
                    
                
                
                distlist=list(plotdata.keys())[1:]
                if datatyp!='discrete':
                    distlist.append('kde')
                #st.write(list(plotdata.keys())[0])
                dist_userselect=st.selectbox("Select the distribution of your choice to generate code. Adjust the num_datapoints parameter to change the number of points to generate",distlist )
                
                if dist_userselect:
                    if dist_userselect=='kde':
                        output=listparamsofdistributions.gencodekde(df_plot['data'],inpvar)
                    else:
                        output=listparamsofdistributions.gencode(dist_userselect, datatyp, df_plot['data'])
                    st.code(output)
                    
                
                        
        
                
               
            
            
           
                    
            
            
       
    else:
        st.warning("Please upload a csv file to proceed")








st.sidebar.subheader("Modeling of time independent data")


data_col,button_col=st.columns(2)
with data_col:
    

    st.markdown('''Please upload a CSV file with the variables to fit and proceed.''')
    uploaded_file = st.file_uploader("Choose a file.")
    
        
    
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

#st.sidebar.image("/home/lekshmi/Downloads/DataFitter/MISC/datafitrlogo.PNG", use_column_width=True)
st.sidebar.image("./MISC/datafitrlogo.PNG", use_column_width=True)
try:
    #uploaded_file = st.file_uploader("Choose a file.")
    if uploaded_file is not None:
        uploaded_file_format=is_CSV(uploaded_file.name)
        if uploaded_file_format  in['csv','CSV'] :
            #uploaded_file = st.file_uploader("Choose a file.")
            #st.write(uploaded_file)
            df1 = pd.read_csv(uploaded_file)
            #st.write(df1)
            
            startcheck=df1.describe(include="all")
            
            
        
    
            
            
        
            IM_uni(uploaded_file, uploaded_file_format,df1)
        #st.sidebar.image("/home/lekshmi/Downloads/DataFitter/MISC/datafitrlogo.PNG", use_column_width=True)
    else:
        st.warning("Please upload a csv file to proceed")
except:
    st.warning("Please upload a csv file which adheres to the format mentioned in the documentation.")
    #uploaded_file = st.file_uploader("Choose a file.")
    
    

        

        



 
    

    

            



    


