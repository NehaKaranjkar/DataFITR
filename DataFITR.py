#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 13:17:27 2022

@author: lekshmi
"""

import streamlit as st
#import pandas as pd
#from io import StringIO
#from PIL import Image
#from IM import modular_IM 
#from IM import kde_silverman_both

def main():
    st.set_page_config(page_title="DataFITR",page_icon="datafitrlogo.png",layout="wide")
    st.image("./MISC/datafitrlogo.PNG",width=450)
    
    #st.title("DataFITR")
    #st.subheader("DataFITR in 10 minutes")
    st.text("DataFITR: A guided tool for Input Modeling for the non expert users")
    st.sidebar.header("DataFITR in 10 minutes")  
    
    
    st.markdown("""<style>.big-font {font-size:20px !important;}</style>""", unsafe_allow_html=True)
    st.markdown('<p class="big-font">DataFITR is a graphical tool for Input Modeling that guides the user in a step by step manner. Input Modeling is a process of fitting a distribution to the input data and determining the parameters of that distribution. DataFITR acts as a tool to aid the process of building a Simulation Digital Twin as Input Modeling is one of the most important steps towards building a Digital Twin.  The tool expects the data to be in a csv format where the column name is the name of the process variable and the rows are the observations. Once a good fit is found the tool produces code that can be inserted by the user</p>', unsafe_allow_html=True)
    st.image('./MISC/sampledata.PNG',width=600,caption="Expected CSV file format")
    IIDtab,Gausmulttab,arbmulttab,TStab,=st.tabs(["Time independent Data","Multivariate Gaussian Data","Arbitrary Multivariate Data","Time Series",])
    
    with IIDtab:
        st.header("Modeling of Time independent Data (Independent and Identical data)")
        st.markdown('<p class="big-font">This is used typically when there is no relation between successive observations of an attribute or a process variable.</p>', unsafe_allow_html=True)
        st.markdown('<p class="big-font">The user supplies the data in a csv format which is observations of a certain random variable and the fitted tool finds the distribution that best fits that data from a range of standard discrete and continuous distributions. This tool supports 97 continuous distributions and 3 discrete distributions. It also allows the user to fit an aribtrary distribution which does not belong to any standard distribution or which is a mixture of 2 or more distributions. It also gives a visualization of the fit of user data with various distributions. Tool also outputs the goodness of fit values for all the distributions that it supports.</p>', unsafe_allow_html=True)
        st.markdown('<p class="big-font">The tool infers the type of data(real valued or integer valued) from it. User can also use this to fit categorical data which are converted into numerical values.</p>', unsafe_allow_html=True)
        

        #st.image(image, caption='Sampledataset.csv')
    with Gausmulttab:
        st.header("Modeling of Multivariate Gaussian Data")
        st.markdown('<p class="big-font">This is used typically when there is a correlation between columns of a dataset. The tool warns the user incase the marginal distributions of the selected columns to fit are not normally distributed. However, it calculates the mean and covariathe data and produces the code to generate multivariate gaussian distribution</p>', unsafe_allow_html=True)
    
    with arbmulttab:
        st.header("Modeling of arbitrary Multivariate Data")
        st.markdown('<p class="big-font">This will be implemented soon.</p>', unsafe_allow_html=True)
        
    with TStab:
        st.header("Modeling of Time Series data")
        st.markdown('<p class="big-font">This is used typically when there is a relation between successive observations of an attribute. This will be implemented soon. </p>', unsafe_allow_html=True)
        
    st.markdown("""---""")
    st.subheader("Authors")
    st.text('Lekshmi P., IIT Goa \nDr.Neha Karanjkar, IIT Goa')
    
    st.markdown("""---""")
    st.subheader("Source code")
    st.markdown("[(Github link of the repo)](https://github.com/LekshmiPremkumar/DataFITR)")
    


if __name__=='__main__':
    main()


    

            



  
    


