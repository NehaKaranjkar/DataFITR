# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:56:14 2023

@author: User
"""

import pandas as pd
import glob,io,os
import streamlit as st
import numpy as np


def ifpathExists(path):
    isExist=os.path.exists(path)
    if isExist:
        return True
    else:
        return False
    
def ifnotemptydir(path):
    dir_list=os.listdir(path)
    if len(dir_list) !=0:
        return True
    else:
        return False
    
st.subheader("Output Summary")
st.markdown("---")
st.sidebar.subheader("View Output Summary")
st.sidebar.image("./MISC/datafitrlogo.PNG", use_column_width=True)

try:
    #st.set_page_config(page_title="DataFITR",page_icon="chart_with_upwards_trend",layout="wide")    
    #path=st.text_input("Enter the path to the result folder",key='pathtoresult')
    
    #st.image("./MISC/datafitrlogo.PNG",width=400)
    st.markdown("This section allows you to view the results of the various fits you have done with the data in a CSV file. Please select a folder to proceed")
    
    
    
    
    path='./output'
    #path='/home/ec2-user/DataFITR/output'
    #st.write(os.listdir(path))
    folder=st.selectbox("Select the folder with output", os.listdir(path))
    
    checkbutton=st.checkbox("Show the results")
    fullpath=path+"/"+folder
    
    
      
    # specifying the path to csv files
    if checkbutton :
        if ifpathExists(fullpath) and ifnotemptydir(fullpath):  
        # csv files in the path
            
            files = glob.glob(fullpath + "/*.csv")
            for i in files:
                checkdf=pd.read_csv(i)
                if checkdf.empty:
                    os.remove(i)
            
            #figs=glob.glob(fullpath + "/*.png")
            files = glob.glob(fullpath + "/*.csv")
            for i in files:
                
                df=pd.read_csv(i)
                #st.write(i)
                #st.write(df[5])
                #st.table(df)
                st.subheader(df['var'][0][:-4])
                #st.write(df['var'][0])
                try:
                    
                    col1,col2=st.columns(2)
                    with col1:
                        st.dataframe(df[['Test','KStest','SSE','Chi squared Test']])
                        
                    with col2:
                        figname=i[:-3]+'png'
                        st.image(figname, width=400)
                except:
                    #st.subheader(df['var'][0])
                    col1,col2=st.columns(2)
                    with col1:
                        st.dataframe(df[["value",'type']])
                    
        else:
            st.write("No such folder/directory is found or folder is empty. Please check")
except:
    st.write("Please upload a csv file with data to fit in the IID fitting tab and fit a variable to view the output")
    
    
   
  
    



#print(df)

#st.dataframe(df)

#f=pd.read_csv(files[0])
#print(pd.read_csv(df.to_csv(index=False)))





#df1=pd.read_csv(io.StringIO(df.to_csv(index=False)),)


#df.to_csv(filename,mode='a',index=False)

