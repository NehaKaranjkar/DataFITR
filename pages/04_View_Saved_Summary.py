# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:56:14 2023

@author: User
"""

import pandas as pd
import glob,io,os
import streamlit as st
import numpy as np
import pickle


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
st.set_page_config(page_title="DataFITR",page_icon="📈",layout="wide")  
st.subheader("Output Summary")
st.markdown("---")
st.sidebar.subheader("View Saved Summary")
st.sidebar.image("./MISC/datafitrlogo.PNG", use_column_width=True)
#st.write("test",st.session_state['foldername'])

st.markdown("This section allows you to view the results of the various column fits you have done with the data in the CSV file.")
    

 
    
    
    
    
checkbutton=st.button("Show the results")


#st.write(st.session_state)  
count=0
if checkbutton :
    for i in st.session_state:
        if i[-5:]=='table':
            count+=1
            st.markdown("""---""")
            st.subheader(i[:-5])
            st.markdown("""---""")
            table,image=st.columns(2)
            with table:
                st.dataframe(st.session_state[i])
            with image:
                st.pyplot(st.session_state[i[:-5]+'image'])
            
            
           
            #st.write(st.session_state)
            if i[:-5]+'pickle' in st.session_state:
                code,file=st.columns(2)
                with code:
                    st.download_button("Click to download the Kernel.pkl file", data=st.session_state[i[:-5]+'pickle'],file_name="kernel.pkl")
                with file:
                    st.code(st.session_state[i[:-5]+'code'])
            else:
                st.code(st.session_state[i[:-5]+'code'])
            #st.markdown("""---""")
        if i[-8:]=='constant':
            count+=1
            st.markdown("""---""")
            st.subheader(i[:-8])
            st.markdown("""---""")
            table,image=st.columns(2)
            with table:
                st.dataframe(st.session_state[i])
            st.info("The data column to fit is a contant value and the value is  "+str(st.session_state[i]['value']))
            #st.markdown("""---""")
        
    
    if count==0: 
        st.write("Please upload a csv file with data to fit in the IID fitting tab and fit a variable to view the output")
     
                      
                    
       
   

    
   
  
    



#print(df)

#st.dataframe(df)

#f=pd.read_csv(files[0])
#print(pd.read_csv(df.to_csv(index=False)))





#df1=pd.read_csv(io.StringIO(df.to_csv(index=False)),)


#df.to_csv(filename,mode='a',index=False)

