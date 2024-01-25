# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:50:15 2023

@author: leksh
"""

def prediction_ARMA(data,theta,phi,sigma,masigma,ar,ma,val):
    val_range=range(val[0]-ar,val[1]-ar)
    sub_theta=[]
    for ind in range(ar):
        sub_theta.append(theta[(ar,ind+1)])
    sub_theta=np.array(sub_theta)
    sub_phi=[]
    for ind in range(ma):
        sub_phi.append(phi[(ma,ind+1)])
    sub_phi=np.array(sub_phi)
    #sigvals=np.array([masigma[i] for i in masigma.keys()[1:]])
    
    if ma==0:
        predar=[]
        predmeanar=[]
        for i in val_range:
            
            xvals=np.array(data[i:i+ar])
            #print(sub_theta)
            #print(xvals)
            #print(xvals*sub_theta)
            pred.append((sum(xvals*sub_theta)))#+np.random.normal(0,sigma[ar])
            #predmean.append(np.mean(xvals)+(sum(xvals*sub_theta))+np.random.normal(0,sigma[ar]))
            #vallist=[i for i in range(val[0]-ar,val[1]-ar)]
        vallist=[i for i in range(val[0],val[1])]
        predict={'ind':vallist,'pred':pred}
    else:
        predar=[]
        predmeanar=[]
        innovvals=[]
        maxlag=(max(ar,ma))
        w={maxlag:0}
        for i in val_range:
            
            xvals=np.array(data[i:i+ar])
            #print(sub_theta)
            #print(xvals)
            #print(xvals*sub_theta)
            predar.append((sum(xvals*sub_theta)))#+np.random.normal(0,sigma[ar])
            #predmean.append(np.mean(xvals)+(sum(xvals*sub_theta))+np.random.normal(0,sigma[ar]))
            #vallist=[i for i in range(val[0]-ar,val[1]-ar)]
        vallist=[i for i in range(val[0],val[1])]
        predict={'ind':vallist,'predar':predar}
        
        #print(predict)
        for i in range(len(predar)):
            #print(i)
            innovvals.append(predar[i]-data[val_range[i]])   
        
        #print("here")
        i=0
        predma=[]
        while(len(predar)-i>=ma):
            
            #print(val_range[-1])
            valforma=np.array(innovvals[i:i+ma])
            
            predma.append((sum(valforma*sub_phi)))
            i+=1
        
        predict['predma']=predma   
        predma=np.array(predma)
        predma=np.array(predar)
        predict['predfinal']=predma +predar 
    #print("hoola",len(predma))
    #print(len(predar))
    #print(predict)
        
    #pred_df=pd.DataFrame(predict)        
    return predict,innovvals,np.array(predict['predma'])+np.array(predict['predar'][:-1])