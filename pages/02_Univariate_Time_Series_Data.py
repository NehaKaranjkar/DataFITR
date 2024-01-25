# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:32:20 2023

@author: leksh
"""

import matplotlib.pyplot as plt
from scipy import stats
import pickle
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import sys,io,os
import streamlit as st
import pandas as pd
import scipy.stats as stats,time
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
#sys.path.append('/home/lekshmi/Downloads/my_app/IM') 
sys.path.append('./IM') 
#import modular_IM
from IM import modular_IM
from IM import kde_silverman_both
from IM import datagenforDataFITR, syntheticdata
#import datagenforDataFITR
#import syntheticdata
#import kde_silverman_both
#fromm IM import wrapper
import math
import re
from IM import listparamsofdistributions
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
#from statsmodels.tsastattools import adfuller
import seaborn as sns
from scipy.stats import pearsonr #to calculate correlation coefficient

def to_csv(df):
    data_csv = df.to_csv(path_or_buf=None, sep=',', index=False) 
    return data_csv
from statsmodels.tsa.stattools import adfuller

def check_stationarity(series):
    # Copied from https://machinelearningmastery.com/time-series-data-stationary-python/

    result = adfuller(series)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        st.info("Data is Stationary")
    else:
        st.warning("Data is Non-stationary. This tool currently only supports modeling Stationary data")



    
def autocorr(data, lag=5):
       c = np.correlate(data, data, 'full')# gets all the crossproductseg:- if data=[1,2,1] then ans=[1*1,1*2++2*1,1*1+2*2+1*1,1*2+2*1,1*1]
       mid = len(c)//2
       acov = c[mid:mid+lag]# from correlate vals we only need from mid- where cross corrln at lag=0, lag=1.... lag midpt
       #print(c)
       acor = acov/acov[0]#no need to divide by n-1 as we are dividing it with acovs itself. n-1 in num and deno will cancel out
       return acor



def main():
    st.set_page_config(page_title="DataFITR",page_icon="📈",layout="wide")
    
    #st.sidebar.subheader("Fitting Time Series Data")
    
    #st.sidebar.image("/home/lekshmi/Downloads/DataFitter/MISC/datafitrlogo.PNG", use_column_width=True)
    st.sidebar.image("./MISC/datafitrlogo.PNG", use_column_width=True)
    st.subheader("Fitting Time Series Data")
    st.markdown("---")
    
    data_col,button_col=st.columns(2)
    with data_col:
        
        datainput_formTS=st.radio("Select a data source",
                ('Upload a CSV','Use a sample synthetic data',))
        if datainput_formTS=='Use a sample synthetic data':
            numdatagen=st.slider("Enter the number of datapoints to generate", 50, 500,400)
       
        else:
            st.write("The expected format is as shown below. ")
            #st.image("./MISC/sampledata.PNG",width=500)
            st.markdown("""
            The tool expects the data to be in a csv format where the column name is the name of the process variable and the rows are the observations.
            """)
    with button_col:
        if datainput_formTS=='Upload a CSV':
            try:
                
                st.markdown('''Please upload a CSV file with the variables to fit and proceed.''')
                uploaded_file = st.file_uploader("Choose a file.",type="csv")
                #st.session_state.data_generatedMAG=False
                if uploaded_file is None:
                    st.warning("Please upload a csv file to proceed")
                    return
                    
                        
                else:
                    dfTS = pd.read_csv(uploaded_file)
                    st.session_state.data_generatedTS=True
                    st.session_state.dataTS=dfTS
                    startcheck=dfTS.describe(include="all")
                    
            except:
                st.warning("Please upload a csv file which adheres to the format mentioned in the documentation.")
                #uploaded_file = st.file_uploader("Choose a file.")
                #IM_uni(dfTS)    
                
        
            
            #IM_uni(dfTS)
    
        else:
            
            #st.session_state.data_generatedTS=False
            if st.button(("Regenerate a sample synthetic data" if "data_generatedTS" in st.session_state else "Generate a sample synthetic data")):
                #dfTS=syntheticdata.univariate_stationary(numdatagen)
                dfTS=datagenforDataFITR.TimeSeriesGen(numdatagen)
                #dfTS=t.generatestationary()
                st.session_state.data_generatedTS=True
                st.session_state.dataTS=dfTS
        
                with st.expander("View raw data"):
                    st.write(dfTS)
                st.download_button('Download generated data as a CSV file', to_csv(dfTS), 'sample_multivariate_gaussian_data.csv', 'text/csv')
                
    if  'data_generatedTS' in st.session_state:
        #st.write(st.session_state.data_generated)
        
        if st.button("Start Exploratory Data Analysis"):
            st.session_state.button_clickedTS=True
    if 'button_clickedTS' in st.session_state:
            datatofitTS=st.session_state['dataTS']
            #st.write(datatofit)    
            IM_TS(datatofitTS)
            
    

    
    
    
def IM_TS(df):   
    
    st.header("Data Exploration")
    inpvar = st.selectbox("data column", df.columns)
    x=df[inpvar]
    
    plot,stn=st.columns(2)
    with plot:
        st.subheader("Plot of the dataset")
        fig,ax=plt.subplots()
        ax.plot(x,c='r')
       
        ax.legend(fontsize=15)
        plt.ylabel(inpvar)
        plt.xlabel('index')
        plt.title('Plot of Time Series data along with predicted data')
        st.pyplot(fig)
    with stn:
        st.caption("Checking whether the data is stationary")
        checkstationarity=st.button("Is the data stationary?")
        if checkstationarity:
            check_stationarity(x)
    lag=int(st.slider("Choose a lag value for acf/pacf plots",min_value=1, max_value=50,value=10))
    fig, ax = plt.subplots(1,2,figsize=(12,5))
    import statsmodels.api as sm
    a1=sm.graphics.tsa.plot_acf(x.values.squeeze(), lags=lag, ax=ax[0])
    ax[0].set_xlabel('Lags')
    ax[0].set_ylabel("Autocorrelation")
    
    a2=sm.graphics.tsa.plot_pacf(x.values.squeeze(), lags=lag, ax=ax[1])
    ax[1].set_xlabel('Lags')
    ax[1].set_ylabel("Partial Autocorrelation")
    
    st.pyplot(fig)
    
    
        
    
    ar_p=int(st.number_input("Enter the p value",min_value=0))
    ma_q=int(st.number_input("Enter the q value",min_value=0))
    

    
    arma_TS = ARIMA(x, order=(ar_p, 0, ma_q)).fit()
    print(arma_TS.params)
    
    ind1=st.slider("Choose the start index",1,len(x),int(len(x)/2),key="ind1")
    ind2=st.slider("Choose the end index",ind1,len(x),len(x),key="ind2")
    #preddf,_,pred=prediction_ARMA(x,t,p,s,sig,ar_p,ma_q,[ind1,ind2])
    
    print(arma_TS.params)
    forecast = arma_TS.predict(start=ind1, end=ind2)
    print(arma_TS.forecast(1))

    fig,ax=plt.subplots()
    ax.plot(x,c='r')
    indices=[i for i in range(ind1,ind2+1)]
    ax.plot(indices,list(forecast),c='black')
    ax.legend(fontsize=15)
    plt.xlabel('time')
    plt.ylabel(inpvar)
    plt.title('Plot of Time Series data along with predicted data')
    st.pyplot(fig)
    nsteps=1
    fitted_model = ARIMA(x, order=(ar_p,0,ma_q)).fit()
    forecast = fitted_model.forecast(steps=1)
    #st.info("The aic and bic values are "+str(fitted_model.aic)+" and "+str(fitted_model.bic))
    fit_TS=st.button("Generate code for Random Variate Generation.")
    #pred_df[['ind','pred']].plot(figsize=(12,8))
    #plt.plot(predmean,c='orange')
    if fit_TS:
        
      
        opt=f"""
        import pandas as pd\nimport numpy as np\nfrom statsmodels.tsa.arima.model import ARIMA\nx=inputdata\n"""
        str_plist="p="+str(ar_p)+"\n"+"q="+str(ma_q)+"\n"+"nsteps=1"+"\n"
        
        
        datastr=f"""fitted_model = ARIMA(x, order=(p,0,q)).fit()\nforecast = fitted_model.forecast(steps=nsteps)\n"""
        
        optoshow=opt+str_plist+datastr
         
        
    
        st.caption("Use the code to generate data")
        st.code(optoshow)
      
        picklfilename="TS"+'pickle'
        st.session_state[picklfilename]=pickle.dumps(optoshow)
    
        
        
   

    
    
    
    
    
    
if __name__=='__main__':
    main()
        
      