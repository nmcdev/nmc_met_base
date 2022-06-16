# _*_ coding: utf-8 _*_

# Copyright (c) 2021 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
 A structural self-organizing map algorithm for weather typing
 refer to:
     https://zenodo.org/record/4437954#.YofDnajP3up
"""

"""
Created on Fri May  8 20:52:27 2020
@author: doan
"""

import os, sys
import numpy as np
from numpy import random as rand
import xarray as xr


#==============================================================================
def strsim(x,y):
    '''
    Parameters
    ----------
    x : input vector 1 (float)
        For comparison 
    y : input vector 2 (float)
        For comparison
    Returns
    -------
    Float
        Structural similarity index (-1 to 1)
    '''
    term1 = 2*x.mean()*y.mean() / (x.mean()**2 + y.mean()**2)
    term2 = 2*np.cov(x.flatten(),y.flatten())[1,0] / (np.var(x)+np.var(y))
    return term1*term2
#==============================================================================

#==============================================================================
def strcnt(x,y):
    '''
    Parameters
    ----------
    x : input vector 1 (float)
        For comparison 
    y : input vector 2 (float)
        For comparison
    Returns
    -------
    Float
        Structural similarity index (-1 to 1)
    '''
    term1 = 1. #2*x.mean()*y.mean() / (x.mean()**2 + y.mean()**2)
    term2 = 2*np.cov(x.flatten(),y.flatten())[1,0] / (np.var(x)+np.var(y))
    return term1*term2
#==============================================================================

#==============================================================================
def S_luminance(x,y): return 2*x.mean()*y.mean() / (x.mean()**2 + y.mean()**2)
#==============================================================================    
#==============================================================================
def S_contrast(x,y): return 2* np.std(x) * np.std(y) / (np.var(x)+np.var(y))
#==============================================================================       
#==============================================================================
def S_structure(x,y): return np.cov(x.flatten(),y.flatten())[1,0] / (np.std(x) * np.std(y))
#==============================================================================
#==============================================================================
def edsim(x,y): return -np.linalg.norm(x - y)
#==============================================================================

sim_func = {'ssim':strsim,
            'ed': edsim,
            'lum':S_luminance,
            'cnt':S_contrast, 
            'str':S_structure, 
            'sc':strcnt}

#==============================================================================
def bmu1d(sample,candidate,method='ed'):
    '''
    Parameters
    ----------
    sample : TYPE
        DESCRIPTION.
    candidate : TYPE
        DESCRIPTION.
    method : TYPE, optional
        DESCRIPTION. The default is 'ed'.

    Returns
    -------
    bmu_1d : TYPE
        DESCRIPTION.
    smu_1d : TYPE
        DESCRIPTION.
    maxv : TYPE
        DESCRIPTION.
    values : TYPE
        DESCRIPTION.

    '''
    if method == 'ssim':
        values = []
        x = sample
        for y in candidate[:]:
            term1 = 2*x.mean()*y.mean() / (x.mean()**2 + y.mean()**2)
            term2 = 2*np.cov(x.flatten(),y.flatten())[1,0] / (np.var(x)+np.var(y))
            values.append(term1*term2)
        values = np.array(values)

    #====================
    if method == 'sc':
        values = []
        x = sample
        for y in candidate[:]:
            term1 = 1. #2*x.mean()*y.mean() / (x.mean()**2 + y.mean()**2)
            term2 = 2*np.cov(x.flatten(),y.flatten())[1,0] / (np.var(x)+np.var(y))
            values.append(term1*term2)
        values = np.array(values)
    #====================
        
    if method == 'ed':  
        sub = candidate - sample
        values = - np.linalg.norm(sub, axis=1)
        
    if method in ['lum', 'cnt', 'str']: 
        values = []
        x = sample
        for y in candidate[:]:
            values.append( sim_func[method](x.flatten(),y.flatten()) )
        values = np.array(values)  
    
    
    maxv = np.max(values) 
    bmu_1d, smu_1d = np.argsort(values)[-1], np.argsort(values)[-2]
    #print(values)
    return bmu_1d, smu_1d, maxv, values
#==============================================================================


#==============================================================================
def som(input2d,n=10,iterate=5000,learnrate=0.1,sim='ed'):
    '''
    Parameters
    ----------
    input2d : float 
        2-dim array, first dimensional represents input samples.
    n : interger
        number of SOM nodes, 
        The default is 10.
    iterate : integer, optional
        Number of iterations. The default is 5000.
    learnrate : float, optional
        Initial learning rate. The default is 0.1.
    sim : string, optional
        Similarity index (two options: 'ed' is nagative ED; 'ssim' is strurural similarity index.)
        The default is 'ed'.

    Returns
    -------
    ovar : dictionary
        DESCRIPTION.

    '''
    # STEP 1:
    
    # Check input data and setup SOM output map
    if len(input2d.shape) !=2:
        print('*** Error: the input data for SOM should have 2 dimensions:\n the 1st is data piece; 2st is vector weights')
        sys.exit()
    
    # Set SOM arrays            
    output2d = np.random.uniform(low=input2d.min(), high=input2d.max(), size=[n, input2d.shape[-1]] )
    index_map = np.arange(n)
    
    lambda_lr, max_lr, lambda_nb = 0.25, learnrate, 0.5
    print('** SOM: ', 'size: ', n, 'sim: ', sim, 'iteration: ', iterate)
    #-------------------------
    life = iterate * lambda_lr
    initial = n*lambda_nb
    
    Lr = np.zeros(iterate)   # leaning rate (each step)
    Nbh = np.zeros(iterate)  # Neiborhood function (each step)
    Bmu_2a = np.zeros( (iterate) ) # best matching unit (each step)
    
    # define output ovar
    ovar = {var: [] for var in ['som_hist','learnrate', 'NbhF', 'ivector', 'bmu', 'bmu_proj']}
    
    # STEP 2: learning
    
    for i in range(iterate):
        # STEP 2-1: select input vector (randomly)
        ind = rand.randint(0, input2d.shape[0])
        data = input2d[ind]
        
        # STEP 2-2: find best matching unit        
        bmu_1d, _, _, _ = bmu1d(data,output2d,method=sim)
        
        Bmu_2a[i] = bmu_1d # save best matching unit
        
        # Step 2-3: Update learning rate and neighborhood function
        dis = index_map - bmu_1d    
        Lr[i] = max_lr * np.exp(-i/life)    
        Nbh[i] = initial*np.exp(-i/life)
        S = np.exp(-dis**2 / (2*Nbh[i]**2))   
        
        # Step 2-4: Update SOM map
        output2d = output2d + Lr[i]* S[:,np.newaxis] * (data - output2d)
        
        # Step 2-5: Save to output var
        if np.mod(i,int(iterate/100)) == 0: 
            
            print('SOM with:', sim, ';   step: ', i)
            
            ovar['som_hist'].append(output2d) # save results
            ovar['learnrate'].append(Lr[i]) # save to learning rate
            ovar['ivector'].append(ind) # save index of input vector
            ovar['bmu'].append(bmu_1d) # save best matching unit
            ovar['NbhF'].append(S) # save neighborhood function


    # Step 3: Output
    ovar['som'] = output2d # save final som map result
    ovar['bmu_proj_fin'] = [bmu1d(data,output2d,method=sim)[0] for ind, data in enumerate(input2d)] # best matching unit at final map for each input vector
    ovar['smu_proj_fin'] = [bmu1d(data,output2d,method=sim)[1] for ind, data in enumerate(input2d)] # second matching unit at final map for each input vector
    ovar['similarity_index'] = sim # save type of similarity index used

    dat = output2d
    if sim == 'ssim':
        values = [ strsim( d1, d2 ) for d1 in dat for d2 in dat ]
    if sim == 'sc':
        values = [ strcnt( d1, d2 ) for d1 in dat for d2 in dat ]
    if sim == 'ed': 
        values = [ - np.linalg.norm(d1 - d2) for d1 in dat for d2 in dat ]
    if sim in ['lum', 'cnt', 'str']:
        values = [ sim_func[sim](d1, d2) for d1 in dat for d2 in dat ]
        
    # save betweenness similarity values (optional)
    ovar['sim_btw'] = np.array(values).reshape(len(dat),len(dat)) #[np.triu_indices(len(dat),k=1)]
    
    # save topological error
    terror = np.abs(np.array(ovar['bmu_proj_fin']) - np.array(ovar['smu_proj_fin'])).mean()
    ovar['topo_error'] = terror


    return ovar
#==============================================================================
