#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 13:17:27 2022

@author: lekshmi
"""

import streamlit as st
import streamlit.components.v1 as components
#import pandas as pd
#from io import StringIO
#from PIL import Image
#from IM import modular_IM 
#from IM import kde_silverman_both

def main():
    st.set_page_config(page_title="DataFITR",page_icon="📈",layout="wide")
    st.image("./MISC/datafitrlogo.PNG",width=450)
    
    #st.title("DataFITR")
    #st.subheader("DataFITR in 10 minutes")
    st.text("DataFITR: A guided tool to interactively fit probability distributions to data for the non expert users")
    st.sidebar.header("DataFITR in 10 minutes")  

    #st.markdown('<a href="http://52.62.118.188:8501" target="_self">Click here to go to DataFITR</a>', unsafe_allow_html=True)
    #st.markdown("[(Click here to go to DataFITR)](http://54.252.148.127:8501)")
    
    
    st.markdown("""<style>.big-font {font-size:20px !important;}</style>""", unsafe_allow_html=True)
    st.markdown('<p class="big-font">DataFITR is a graphical tool for Input Modeling that guides the user in a step by step manner. Input Modeling is a process of fitting a distribution to the input data and determining the parameters of that distribution. DataFITR acts as a tool to aid the process of building a Simulation Digital Twin as Input Modeling is one of the most important steps towards building a Digital Twin.  The tool expects the data to be in a csv format where the column name is the name of the process variable and the rows are the observations. Once a good fit is found the tool produces code that can be inserted by the user into the simulation model</p>', unsafe_allow_html=True)
    st.image('./MISC/sampledata.PNG',width=600,caption="Expected CSV file format")
    st.sidebar.markdown('''
    # Sections
    - [Time Independent Data](#time-independent-data")
    - [Gaussian Multivariate Data](#gaussian-multivariate-data)
    - [Arbitrary Multivariate Data & Time Series Data ](#arbitrary-multivariate-data)
    
    ''', unsafe_allow_html=True)
    
    
    
    st.subheader("Time Independent Data (Independent and Identical data) ")
    #st.subheader("Modeling of Time independent Data ")
    st.markdown('<p class="big-font">Time Independent Data is the one for which there is no relation between successive observations of an attribute or a process variable.</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font">The user supplies the data in a csv format which is observations of a certain random variable and the fitted tool finds the distribution that best fits that data from a range of standard discrete and continuous distributions. This tool supports 97 continuous distributions and 3 discrete distributions. It also allows the user to fit an aribtrary distribution which does not belong to any standard distribution or which is a mixture of 2 or more distributions. It also gives a visualization of the fit of user data with various distributions. Tool also outputs the goodness of fit values for all the distributions that it supports.</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font">The tool infers the type of data(real valued or integer valued) from it. User can also use this to fit categorical data which are converted into numerical values.</p>', unsafe_allow_html=True)
    st.markdown("[Click here for fitting Time Independent Data](https://datafitr.streamlit.app/Fit_Time_Independent_Data)")
    

        #st.image(image, caption='Sampledataset.csv')
    st.subheader("Gaussian Multivariate Data")
    #st.header("Modeling of Multivariate Gaussian Data")
    st.markdown('<p class="big-font">Modeller can use this if the data to fit is multivariate and each of its components follow a univariate normal distribution and there is a relation between columns of a dataset. The tool prints the Goodness of fit test results of the columns selected with that of a normal distribution. It calculates the mean and covariance of the data and produces the code to generate multivariate gaussian distribution</p>', unsafe_allow_html=True)
    st.markdown("[Click here for fitting Gaussian Multivariate Data](https://datafitr.streamlit.app/Fit_Gaussian_Multivariate_Data)")
    st.subheader("Arbitrary Multivariate Data & Time Series Data")
    #st.header("Modeling of arbitrary Multivariate Data")
    st.markdown('<p class="big-font">This will be implemented soon.</p>', unsafe_allow_html=True)
        
    
        
    st.markdown("""---""")
    st.subheader("Authors")
    st.text('Lekshmi P., IIT Goa \nDr.Neha Karanjkar, IIT Goa')
    
    st.markdown("""---""")
    st.subheader("Source code")
    st.markdown("[(Github link of the repo)](https://github.com/NehaKaranjkar/DataFITR)")
    
     # embed streamlit docs in a streamlit app
    #components.iframe("http://52.62.118.188:8501")
    



if __name__=='__main__':
    main()


    

            



  
    


