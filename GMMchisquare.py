#!/usr/bin/env python
# coding: utf-8

# In[1]:


###############################GMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM###########################
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib as plt
import seaborn as sns; sns.set(color_codes=True)
import sklearn.preprocessing as sklp
import sklearn.feature_selection as sklf
import sklearn.metrics as sklm
import matplotlib.pyplot as plt
import math
from tqdm import tqdm_notebook as tqdm
import time
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt
from scipy.stats import norm
import itertools as itert
from numpy import *
import matplotlib.patches as mpatches
import matplotlib.ticker as tick
import matplotlib.lines as mlines
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

class GMMchisquare():

    get_ipython().run_line_magic('matplotlib', 'inline')

    def dynamic_binning(observed,binedges,threshold=5,final=True):
        """ This function is used to dynamically add samples to each bin to ensure maximize and ensure each bin has at least >=5 samples

        :param observed: This is the input observed number of bins
        :param binedges: This is the input bin edges for each bin
        :param threshold: This is the number of samples expected for each bin
        :param final: This is used to initialize the function

        """    

        continued = 1
        x=0 #dynamically update range
        orgi = observed.copy()

        while x < len(observed):
            try:
                restart = True #loop until larger than threshold
                while restart:

                    if observed[x]<threshold:
                        observed[x] = observed[x]+observed[x+1]
                        observed = np.delete(observed,x+1)
                        binedges = np.delete(binedges,x+1)
                    else:
                        restart = False
                x+=1
            except: #sometimes you will get bins with >threshold in the very last bin 
                if observed[-1] <threshold:

                    observed[-1] = observed[-1]+observed[-2]
                    observed=np.delete(observed,-2)
                    binedges=np.delete(binedges,-2)

        if len(orgi) == len(observed):
            continued = 0                

        if final == False:
            largedata = np.arange(len(observed))[np.in1d(observed, orgi)] #get overlapping index

            try:
                if min(largedata) != 0 or max(largedata)+1 != len(observed): #check whether it is spanning the entire dataset

                    #expand the space where there is no tail problem to the number of bins using mann and wald and leave the rest
                    if len(largedata) == 1:
                        nums = int(1.88*observed[largedata[0]]**(2/5)) #mann and wald 
                        x = largedata[0]
                        newbins = np.linspace(binedges[largedata[0]],binedges[largedata[0]+1], num=nums)
                        binedges = np.delete(binedges,(largedata[0],largedata[0]+2))
                        binedges = np.sort(np.concatenate((binedges,newbins)))
                    else:
                        nums = int(1.88*sum(observed[min(largedata):max(largedata)+1])**(2/5)) #mann and wald 
                        newbins = np.linspace(binedges[min(largedata)],binedges[max(largedata)+1], num=nums)
                        binedges = np.delete(binedges,np.arange(min(largedata),max(largedata)+2))
                        binedges = np.sort(np.concatenate((binedges,newbins)))
            except:
                pass #if they are the same during the middle stage

        return observed,binedges,continued

    #####################################################################################################################
    def probe_filter(datainputnorm,log2transform=False, filt=0,threshold_filter = 0.01,variance_filter = 0.0125):

        """ This function is used to filter probes or genes that are expressed under the pre-defined background threshold. Users
        can also define the variance filter to exclude probes that has most expression level below the background threshold with
        a few escaping expression level that are slightly above the threshold.

        :param datainputnorm: Input dataframe with genes (row) x samples (columns)
        :param log2transform: Default set to False. This is used to indicate whether to log2-transform input data for filtering 
        :param filt: Default set to 0. This is the input level of background threshold
        :param threshold_filter: Default set to 0.01. This is the percentage of the total number of samples considered for testing 
            the number of 'escaped sample' with expression level slightly higher than bakcgroudn threshold
        :param variance_filter: Default set to 0.0125. This is the quantile of the overall variance that will be used as the variance
            for filtering 'escaped sample'

        Returns the normal data with probes filtered out 

        """
        if log2transform == True:
            ip_pseudo = datainputnorm.replace(0, np.NAN) #pseudocount so that we do not ignore gene with 0 expression
            datainput = np.log2(ip_pseudo) 

            data = datainput.copy()
    #         data.replace(np.nan, filt-1,inplace=True)
            aa = []
            var = []
            for a in tqdm(range(0,len(data))):
                percent = sum(data.iloc[a,:] >= filt)
                aa.append(percent)

            aap = pd.DataFrame({'threshold':aa}, index = data.index)
            aap = aap[aap['threshold']>math.ceil(threshold_filter*len(datainput.T))] #remove samples with all sample below expression threshold

            datainput = datainput.loc[aap.index]
            datainputnorm = datainputnorm.loc[aap.index]

            for a in tqdm(range(0,len(datainput))):
                varss = np.var(datainput.iloc[a,:])
                var.append(varss)

            varp = pd.DataFrame({'variance':var}, index = datainput.index)
            varp = varp[varp['variance']<(max(var)-min(var))*variance_filter] #remove sample with low variance

            finalpd = pd.concat([varp, aap],axis=1,join='inner') 

            data = datainputnorm.drop(finalpd.index) #remove sample with low variance and below threshold filter
        else:
            data = datainputnorm.copy()
            data.replace(np.nan, filt-1,inplace=True)
            aa = []
            var = []
            for a in tqdm(range(0,len(data))):
                percent = sum(data.iloc[a,:] >= filt)
                aa.append(percent)

            aap = pd.DataFrame({'threshold':aa}, index = data.index)
            aap = aap[aap['threshold']>math.ceil(threshold_filter*len(datainputnorm.T))] #remove samples with all sample below expression threshold

            datainputnorm = datainputnorm.loc[aap.index]

            for a in tqdm(range(0,len(datainputnorm))):
                varss = np.var(datainputnorm.iloc[a,:])
                var.append(varss)

            varp = pd.DataFrame({'variance':var}, index = datainputnorm.index)
            varp = varp[varp['variance']<(max(var)-min(var))*variance_filter] #remove sample with low variance

            finalpd = pd.concat([varp, aap],axis=1,join='inner') 

            data = datainputnorm.drop(finalpd.index) #remove sample with low variance and below threshold filter
        return data
    ###############################GMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM###########################


    mpl.rcParams['figure.dpi'] = 300
    warnings.filterwarnings("ignore")

    get_ipython().run_line_magic('matplotlib', 'inline')


    def GMM_modelingt(ID,input_datanormal,log2transform = True, dynamic_binning_s=True, tune_factor=0.99, verbosity = False, graphs = True, 
                      farinto = 0.1, calc_back = False, calc_backpara = False,filt=0,meanf = 0, stdf = 0, chisquaremethod = True, Single_tail_validation = True, 
                      find_target=False, cell_lines=[]):
        """ This function is the Gaussian Mixture Modeling using the Chi-squared Fit protocol to subcategorize data distribution

        :param ID: Probe or gene name
        :param input_datanormal: Input dataframe with genes (row) x samples (columns)
        :param log2transform: Default set to True. This is used to indicate whether to log2-transform is required for input data
        :param dynamic_binning_s: Default set to True. This calls the dynamic binning function to ensure each bin contain >= 5 samples
        :param tune_factor: Default set to 0.99. This is multiplied to each successive chi-square value to ensure the next is lower than the previous
        :param verbosity: Default set to False. True prints out function output of the chi-squared fit protocol
        :param graphs: Default set to True. This gives user control of whether they need to output the resulting graphs 
        :param farinto: Default set to 0.1. This parameter is used to allow the chi-square fit protocol to take off datapoints a little beyond the 
            tail to make sure it takes all possibilities into acount
        :param calc_back: Default set to False. True means ignore everything and calculate background threshold
        :param calc_backpara: Default set to False. True means taking background threshold filt, meanf, stdf into account
        :param meanf: Default set to 0. This is the input mean of the background distribution
        :param filt: Default set to 0. This is the input level of background threshold
        :param stdf: Default set to 0. This is the standard deviation of the background distribution
        :param chisquaremethod: Default set to True. False turns of the chi-squared fit protocol
        :param Single_tail_validation: Default set to True. False turns identifying tail problem off
        :param find_target: Default set to False. Users can input sample names desired to be graphed by making this parameter True
        :param cell_lines: Default set to []. Users input sample names if find_target is True

        If calc_back = True: Returns means, np.sqart(covars), threshold
        If calc_back = False: Returns means, covariance, threshold, classification, classified subgroups of each sample, chi-square value

        Classification annotation:
        1: Unimodal distribution
        2: Bimodal distribution
        21: Unimodal distribution + tail
        3: Bimodal distribution + tail
        20: Biomdal distribution with chi-square fit protocol failed to fit 

        """
        if calc_back == True:
            idf = input_datanormal.values.flatten()
            datanorm = idf[~np.isnan(idf)].reshape(-1,1)

            if log2transform == True:
                #log2 transform
                input_pseudo = input_datanormal.replace(0, np.NAN) #pseudocount so that we do not ignore gene with 0 expression
                datalog = np.log2(input_pseudo) 

                idflog = datalog.values.flatten()
                data = idflog[~np.isnan(idflog)].reshape(-1,1)

        else:
            datanorm = input_datanormal.loc[ID].dropna().values.reshape(-1,1) 
            input_datanorm = pd.DataFrame(input_datanormal.loc[ID].dropna()).T
            input_datanormcat = pd.DataFrame(input_datanormal.loc[ID]).T #used for categories cuz we need NA to count
            if log2transform == True:
                #log2 transform
                input_pseudo = input_datanorm.replace(0, np.NAN) #pseudocount so that we do not ignore gene with 0 expression
                input_data = pd.DataFrame(np.log2(input_datanormal.loc[ID].dropna().astype(np.float64))).T

                input_pseudo = input_pseudo.loc[ID].dropna().values.reshape(-1,1).astype(np.float64) 

                data = np.log2(input_pseudo) 



        if log2transform==True:
            #GMM parameters
            tol = 1e-8 #convergence threshold
            n_mod = 3 #number of models for BIC comparison

            #BIC
            n_components = np.arange(1,n_mod)
            models = [GMM(n,tol=tol,random_state=41).fit(data) #state=41 to make sure it doesn't always start random
                      for n in n_components]

            #BIC and AIC for validating optimal n
            BIC = [m.bic(data) for m in models]

            #use BIC as our standard due to limited data 
            n_comp = np.argmin(BIC)+1

            # use guassian mixture modeling to model bimodal distribution
            gmm = GMM(n_components =n_comp,tol=tol,random_state=41).fit(data)

            #sort for cutoff 
            x_axis2 = np.sort(np.array([b for a in data for b in a]))


            #recreate logscaled distribution
            xs = np.linspace(np.min(data),np.max(data), num=499) #recreate normal distribution
            xxs = np.linspace(np.min(data),np.max(data), num=500)
            dx =np.diff(xxs) #calculate the integral value dx

            #recreate normal distribution with fitted GM
            means = gmm.means_
            covars = gmm.covariances_
            weights = gmm.weights_
            if n_comp > 1:
                yss = weights[0]*stats.multivariate_normal.pdf(xs, mean = means[0][0],cov = covars[0][0])
                yss2 = weights[1]*stats.multivariate_normal.pdf(xs, mean = means[1][0],cov = covars[1][0])
                expected = yss*dx*len(data)
                expectedt = yss2*dx*len(data)
                expectedboth = (yss+yss2)*dx*len(data)
            else: 
                yss = stats.multivariate_normal.pdf(xs, mean = means[0][0],cov = covars[0][0])
                expected = yss*dx*len(data)

            #finding out groups
            groups = [np.argmax(a) for a in gmm.predict_proba(x_axis2.reshape(-1,1)).round(0)] #find which group each data point is under

            #bimodality 

            roll = groups != np.roll(groups,1)
            roll[0] = False  #roll will make the first variable True but we do not want that
            group_index = [x for x in np.where(roll)]
            group_div = x_axis2[group_index]

            if len(group_div) == 0: #little distribution under big distribution (inclusion) is not the type of bimodality this algo is for return classif, categories, chis[-1]
                n_comp =1 

        #     ------------------------------------measure chi-square of fitted GMM-------------------------------------------------
        #     determin number of bin using dynamic binning
            datas = np.array([[a] for a in np.sort(np.array([b for a in data for b in a]))]) 
            nums=int(1.88*len(datas)**(2/5)) #mann and wald 

            observed, bin_edges = np.histogram(data, bins=np.linspace(np.min(data),np.max(data), num=nums), density=False)

            if dynamic_binning_s == True:
                golddof = len(observed)
                lenn = 0
                while lenn < golddof: #Loop until you have the ssame dof or unimprovable

                    observed, bin_edges,con = dynamic_binning(observed, bin_edges,final=False)

                    observed, what = np.histogram(data, bins=bin_edges)

                    observed, bin_edges, con = dynamic_binning(observed, bin_edges)

                    observed, what = np.histogram(data, bins=bin_edges)

                    if con == 0:
                        break

                    lenn = len(observed)

            #fit with a single GM
            xschi = bin_edges[1:] #recreate normal distribution using dynamically binned edges
            dxchi = np.diff(bin_edges) #calculate the integral value dx

            if n_comp > 1:
                ysschi = weights[0]*stats.multivariate_normal.pdf(xschi, mean = means[0][0],cov = covars[0][0])
                yss2chi = weights[1]*stats.multivariate_normal.pdf(xschi, mean = means[1][0],cov = covars[1][0])
                expectedchi = (ysschi+yss2chi)*dxchi*len(data)
            else: 
                ysschi = stats.multivariate_normal.pdf(xschi, mean = means[0][0],cov = covars[0][0])
                expectedchi = ysschi*dxchi*len(data)

            #calculate dof
            dof = len(observed)-1


            #calculate chi-square
            arrchi= np.array([[x,y] for x,y in zip(observed,expectedchi)])

            cc = sum([(x[0]-x[1])**2./x[1] for x in arrchi])
            pp = 1-stats.chi2.cdf(cc,dof)

            bimodalc = cc

            bins = 10 #used for histogram and chi-square
            count=0
            counting = 0 
            conditions = [5] #default with condition 5 meaning not fixed
            condition = 0 #default for those that doesn't go through any chi-square method
            chis = []
            rem = [0] #default with condition 5 meaning not fixed so removes nothing
            BICs = []
            cclemon = [] #record chisquare
            biclemon = [] #record BIC
            realbic = []
            nums=int(1.88*len(data)**(2/5)) #mann and wald

        #     -----------------------------IF CHI-SQUARE VALUE > DOF WITH A SINGLE TAIL, RUN CHI-SQUARE METHOD-------
            if Single_tail_validation == True:
                if cc > dof and n_comp == 2 and len(group_div) < 2:
                    if max(weights)/min(weights) > 3: #IS THERE A TAIL????
                        if verbosity == True:
                            print('Single tail problem that may not be normally distributed, run chi-square method')
                        tail1 = [num for num in data if num <= group_div]
                        tail2 = [num for num in data if num >= group_div]


                        #deteremine which is the tail to begin trimming and fitting
                        if len(tail1) > len(tail2):
                            thetail = tail2
                            direct = 0
                        else:
                            thetail = tail1
                            direct = 1

                        #determine the number of max datapoints to take off
                        tailmax = math.ceil(len(thetail))
                        if tailmax < 0.2*len(data):
                            tailmax = math.ceil(tailmax+farinto*tailmax)
                        if verbosity == True:
                            print('Number of datapoints in tail:', tailmax)
                        datas = np.array([[a] for a in np.sort(np.array([b for a in data for b in a]))]) #sort data

                        if len(thetail) >0: #if there are more than 1 datapoints in the tails then cut

                            while True:
                                for x in range(1,tailmax):

                                    count=0 #re-zero counter to 0 every loop
                                    if direct == 0:
                                        datamut = datas[:-x]
                                    if direct == 1:
                                        datamut = datas[x:]

                                    nums=int(1.88*len(datamut)**(2/5)) #mann and wald

                                    #BIC
                                    models1 = [GMM(n,tol=tol,random_state=41).fit(datamut)
                                              for n in n_components]

                                    #BIC and AIC for validating optimal n
                                    BIC1 = [m.bic(datamut) for m in models1]

                                    #use BIC as our standard due to limited data 
                                    n_comp1 = np.argmin(BIC1)+1

                                    #dynamically making sure we result in the same number of bins
                                    observed, bin_edges = np.histogram(datamut, 
                                                                       bins=np.linspace(np.min(datamut),np.max(datamut), num=nums), density=False)
                                    if dynamic_binning_s == True:
                                        golddof = len(observed)
                                        lenn = 0
                                        while lenn < golddof:

                                            observed, bin_edges,con = dynamic_binning(observed, bin_edges,final=False)

                                            observed, what = np.histogram(datamut, bins=bin_edges)

                                            observed, bin_edges,con = dynamic_binning(observed, bin_edges)

                                            observed, what = np.histogram(datamut, bins=bin_edges)

                                            if con == 0:
                                                break

                                            lenn = len(observed)


                                    #fit with a single GM
                                    gmmmut = GMM(n_components =1,tol=tol,random_state=41).fit(datamut)
                                    xs = bin_edges[1:] #recreate normal distribution using dynamically binned edges
                                    dx =np.diff(bin_edges) #calculate the integral value dx

                                    #recreate normal distribution with fitted GM
                                    means = gmmmut.means_
                                    covars = gmmmut.covariances_
                                    ys = stats.multivariate_normal.pdf(xs, mean = means[0][0],cov = covars[0][0])
                                    expected = dx*ys*np.sum(observed)


                                    #calculate dof
                                    dof = len(observed)-1

                                    #calculate chi-square
                                    arr= np.array([[x,y] for x,y in zip(observed,expected)])
                                    c = sum([(x[0]-x[1])**2./x[1] for x in arr])
                                    p = 1-stats.chi2.cdf(c,dof)

                                    #fit with two GMs
                                    gmmmut2 = GMM(n_components =2,tol=tol,random_state=41).fit(datamut)

                                    #figure where there are two groups still or not
                                    g1 = [np.argmax(a) for a in gmmmut2.predict_proba(datamut.reshape(-1,1)).round(0)] #find which group each data point is under
                                    #bimodality 
                                    r1 = g1 != np.roll(g1,1)
                                    r1[0] = False  #roll will make the first variable True but we do not want that
                                    gi1 = datamut[[x for x in np.where(r1)]]

                                    #recreate normal distribution with 2 fitted GM
                                    weights2 = gmmmut2.weights_
                                    means2 = gmmmut2.means_
                                    covars2 = gmmmut2.covariances_

                                    yss = weights2[0]*stats.multivariate_normal.pdf(xs, mean = means2[0][0],cov = covars2[0][0])
                                    yss2 = weights2[1]*stats.multivariate_normal.pdf(xs, mean = means2[1][0],cov = covars2[1][0])
                                    expected2 = (yss+yss2)*dx*np.sum(observed)

                                    #calculate chi-square
                                    arr2= np.array([[x,y] for x,y in zip(observed,expected2)])
                                    c2 = sum([(x[0]-x[1])**2./x[1] for x in arr2])
                                    p2 = 1-stats.chi2.cdf(c2,dof)


                                    #reset xs
                                    xs = np.linspace(np.min(data),np.max(data), num=499) #recreate normal distribution
                                    xxs = np.linspace(np.min(data),np.max(data), num=500)
                                    dx =np.diff(xxs) #calculate the integral value dx

                                    #is it better than the original fit?
                                    if counting == 0:
                                        #degrees of freedom factor
                                        ctf = round(cc*tune_factor,2) 
                                        if x == 1:
                                            cclemon.append([ctf,ctf])
                                            biclemon.append([n_comp])
                                            realbic.append(BIC1)
                                            chis.append(ctf)
                                            rem.append(1)
                                    else:
                                        #chisquare tunning factor
                                        ctf = round(chis[-1]*tune_factor,2)
                                    if n_comp1 == 1:
                                        if c < ctf:
                                            if verbosity == True:
                                                print('Removed %d datanormpoints and fitting with 1 Gaussian Mixture with Chi-square value %s <= %s' %(x,np.round(c,2),ctf))
                                            count=1
                                            fc = np.round(c,2)

                                    if n_comp1 == 2:
                                        if len(gi1)<2:
                                            if c2 <= ctf/2:
                                                if verbosity == True:
                                                    print('Removed %d datanormpoints and fitting with 2 Gaussian Mixture with Chi-square value %s <= %s' %(x,np.round(c2,2),ctf))
                                                count=2
                                                fc = np.round(c2,2)

                                    if count >0: #only begin recording when it fits under one of the conditions above  
                                        if direct == 1:
                                            conditions.append(count)
                                            chis.append(fc)
                                            counting+=1
                                            rem.append(x)
                                            BICs.append(BIC1)
                                        if direct == 0:
                                            conditions.append(count+2) #if direct == 0 means we are trimming from the right so add 2 for conditions and count
                                            chis.append(fc)
                                            counting+=1
                                            rem.append(x)
                                            BICs.append(BIC1)
                                            if n_comp1 ==1:
                                                count = 3
                                            if n_comp1 ==2:
                                                count = 4

                                    cclemon.append([c,c2])
                                    biclemon.append([n_comp1])
                                    realbic.append(BIC1)

                                if count ==0 :
                                    count=5 #fifth condition where it did not fix
                                    break

                            condition = conditions[-1]
                            remove = rem[-1]

                            if condition == 1:
                                datamut = datas[remove:]
                                n_comp=1
                                gmm = GMM(n_components =1,tol=tol,random_state=41).fit(datamut)
                            elif condition == 2:
                                datamut = datas[remove:]
                                n_comp=2
                                gmm = GMM(n_components =2,tol=tol,random_state=41).fit(datamut)
                            elif condition == 3:
                                datamut = datas[:-remove]
                                n_comp=1
                                gmm = GMM(n_components =1,tol=tol,random_state=41).fit(datamut)
                            elif condition == 4:    
                                datamut = datas[:-remove]
                                n_comp=2
                                gmm = GMM(n_components =2,tol=tol,random_state=41).fit(datamut)
                            elif condition == 5:
                                n_comp=2
                                gmm = GMM(n_components =2,tol=tol,random_state=41).fit(data)
                                if verbosity == True:
                                    print('Chi-square Method Did Not Help the Tail Problem')

                            #BIC
                            if condition == 1 or condition ==2 or condition == 3 or condition == 4:
                                BIC = BICs[-1]
                            else:
                                n_components = np.arange(1,n_mod)
                                models = [GMM(n,tol=tol,random_state=41).fit(data)
                                          for n in n_components]

                                #BIC and AIC for validating optimal n
                                BIC = [m.bic(data) for m in models]

                                #optimal n_comp
                                n_comp = np.argmin(BIC)+1


                            # use guassian mixture modeling to model bimodal distribution
                            if condition == 1 or condition  ==2 or condition == 3 or condition ==4:
                                x_axis2 = np.sort(np.array([b for a in datamut for b in a]))
                            else:
                                x_axis2 = np.sort(np.array([b for a in data for b in a]))

                            #recreate normal distribution with fitted GM
                            means = gmm.means_
                            covars = gmm.covariances_
                            weights = gmm.weights_

                            if n_comp > 1:
                                yss = weights[0]*stats.multivariate_normal.pdf(xs, mean = means[0][0],cov = covars[0][0])
                                yss2 = weights[1]*stats.multivariate_normal.pdf(xs, mean = means[1][0],cov = covars[1][0])
                                expected = yss*dx*len(data)
                                expectedt = yss2*dx*len(data)
                                expectedboth = (yss+yss2)*dx*len(data)
                            else: 
                                yss = stats.multivariate_normal.pdf(xs, mean = means[0][0],cov = covars[0][0])
                                expected = yss*dx*len(data)

                            #finding out groups
                            groups = [np.argmax(a) for a in gmm.predict_proba(x_axis2.reshape(-1,1)).round(0)] #find which group each data point is under

                            #bimodality 

                            roll = groups != np.roll(groups,1)
                            roll[0] = False  #roll will make the first variable True but we do not want that
                            group_index = [x for x in np.where(roll)]
                            group_div = x_axis2[group_index]
                        else:
                            n_comp=1

        #     ----------------------------IF TAIL PROBELM RUN ------------------------------------------------------------
            if chisquaremethod == True:
                if len(group_div) > 1:
                    if verbosity == True:
                        print('Rerunning GMM with Chi-square Method to fix tail problem')

                    tail1 = [num for num in data if num <= np.min(group_div)]
                    tail2 = [num for num in data if num >= np.max(group_div)]



                    chiv=[]
                    pv=[]
                    chiv2=[]
                    pv2=[]
                    chiv3=[]
                    pv3=[]
                    chiv4=[]
                    pv4=[]
                    BICs=[]
                    gs = []
                    gs2 = []


                    #determine the number of max datapoints to take off
                    xup = math.ceil(len(tail1))
                    yup = math.ceil(len(tail2))

                    #add to the boundary if the tail is very small to allow more data taken off for validation
                    if xup < 0.2*len(data):
                        xup = math.ceil(xup+2+xup*farinto)
                    if yup < 0.2*len(data):
                        yup = math.ceil(yup+2+yup*farinto) #plus two to avoid 0 len tails

                    if verbosity == True:
                        print('Number of datapoints in the right tail:', xup)
                        print('Number of datapoints in the left tail:', yup)



                    while True:
                        for x, y in itert.zip_longest(range(1,xup),range(1,yup)): 


                            datas = np.array([[a] for a in np.sort(np.array([b for a in data for b in a]))]) #sort data so we are taking off the right tail
                            count=0 #re-zero counter to 0 every loop
                            try:
                                datamut = datas[x:] #tail1 data
                                datamut2 = datas[:-y] #tail2 data

                            except:
                                pass

                            nums=int(1.88*len(datamut)**(2/5)) #mann and wald

                            #BIC
                            models1 = [GMM(n,tol=tol,random_state=41).fit(datamut)
                                      for n in n_components]

                            #BIC and AIC for validating optimal n
                            BIC1 = [m.bic(datamut) for m in models1]

                            #use BIC as our standard due to limited data 
                            n_comp1 = np.argmin(BIC1)+1

                        # ---------------------------------tail1----------------------------------------------------

                            observed, bin_edges = np.histogram(datamut, 
                                                               bins=np.linspace(np.min(datamut),np.max(datamut), num=nums), density=False)
                            if dynamic_binning_s == True:
                                golddof = len(observed)
                                lenn=0
                                while lenn < golddof:
                                    observed, bin_edges,con = dynamic_binning(observed, bin_edges,final=False)

                                    observed, what = np.histogram(datamut, bins=bin_edges)

                                    observed, bin_edges,con = dynamic_binning(observed, bin_edges)

                                    observed, what = np.histogram(datamut, bins=bin_edges)

                                    if con == 0:
                                        break

                                    lenn=len(observed)
                            #fit with a single GM
                            gmmmut = GMM(n_components =1,tol=tol,random_state=41).fit(datamut)
                            xs = bin_edges[1:] #recreate normal distribution using dynamically binned edges
                            dx =np.diff(bin_edges) #calculate the integral value dx

                            #recreate normal distribution with fitted GM
                            means = gmmmut.means_
                            covars = gmmmut.covariances_
                            ys = stats.multivariate_normal.pdf(xs, mean = means[0][0],cov = covars[0][0])
                            expected = dx*ys*np.sum(observed)


                            #calculate dof
                            dof = len(observed)-1

                            #calculate chi-square
                            arr= np.array([[x,y] for x,y in zip(observed,expected)])
                            c = sum([(x[0]-x[1])**2./x[1] for x in arr])
                            p = 1-stats.chi2.cdf(c,dof)
                            chiv.append(c),pv.append(p)

                            #fit with two GMs
                            gmmmut2 = GMM(n_components =2,tol=tol,random_state=41).fit(datamut)

                            #figure where there are two groups still or not
                            g1 = [np.argmax(a) for a in gmmmut2.predict_proba(datamut.reshape(-1,1)).round(0)] #find which group each data point is under
                            #bimodality 
                            r1 = g1 != np.roll(g1,1)
                            r1[0] = False  #roll will make the first variable True but we do not want that
                            gi1 = datamut[[x for x in np.where(r1)]]
                            gs.append(len(gi1))

                            #recreate normal distribution with 2 fitted GM
                            weights2 = gmmmut2.weights_
                            means2 = gmmmut2.means_
                            covars2 = gmmmut2.covariances_
                            yss = weights2[0]*stats.multivariate_normal.pdf(xs, mean = means2[0][0],cov = covars2[0][0])
                            yss2 = weights2[1]*stats.multivariate_normal.pdf(xs, mean = means2[1][0],cov = covars2[1][0])
                            expected2 = (yss+yss2)*dx*np.sum(observed)

                            #calculate chi-square
                            arr2= np.array([[x,y] for x,y in zip(observed,expected2)])
                            c2 = sum([(x[0]-x[1])**2./x[1] for x in arr2])
                            p2 = 1-stats.chi2.cdf(c2,dof)
                            chiv2.append(c2),pv2.append(p2)


                            #BIC
                            models2 = [GMM(n,tol=tol,random_state=41).fit(datamut2)
                                      for n in n_components]

                            #BIC and AIC for validating optimal n
                            BIC2 = [m.bic(datamut2) for m in models2]

                            #use BIC as our standard due to limited data 
                            n_comp2 = np.argmin(BIC2)+1
                        # ---------------------------------tail2----------------------------------------------------
                            observed2, bin_edges2 = np.histogram(datamut2, 
                                                               bins=np.linspace(np.min(datamut2),np.max(datamut2), num=nums), density=False)

                            if dynamic_binning_s == True:
                                golddof = len(observed2)
                                lenn=0
                                while len(observed2) < golddof:
                                    observed2, bin_edges2,con = dynamic_binning(observed2, bin_edges2,final=False)

                                    observed2, what = np.histogram(datamut2, bins=bin_edges2)

                                    observed2, bin_edges2,con = dynamic_binning(observed2, bin_edges2)

                                    observed2, what = np.histogram(datamut2, bins=bin_edges2)

                                    if con ==0:
                                        lenn=len(observed2)

                            #fit with a single GM
                            gmmmut3 = GMM(n_components =1,tol=tol,random_state=41).fit(datamut2)
                            xs2 = bin_edges2[1:] #recreate normal distribution
                            dx2 =np.diff(bin_edges2) #calculate the integral value dx

                            #recreate normal distribution with fitted GM cutting from tail2
                            meanso = gmmmut3.means_
                            covarso = gmmmut3.covariances_
                            yso = stats.multivariate_normal.pdf(xs2, mean = meanso[0][0],cov = covarso[0][0])
                            expectedo = dx2*yso*np.sum(observed2)

                            #calculate chi-square
                            arro= np.array([[x,y] for x,y in zip(observed2,expectedo)])
                            c3 = sum([(x[0]-x[1])**2./x[1] for x in arro])
                            p3 = 1-stats.chi2.cdf(c3,dof)
                            chiv3.append(c3),pv3.append(p3)
                            #fit with two GMs
                            gmmmut4 = GMM(n_components =2,tol=tol,random_state=41).fit(datamut2)

                            #figure where there are two groups still or not
                            g2 = [np.argmax(a) for a in gmmmut4.predict_proba(datamut2.reshape(-1,1)).round(0)] #find which group each data point is under
                            #bimodality 
                            r2 = g2!=np.roll(g2,1)
                            r2[0] = False  #roll will make the first variable True but we do not want that
                            gi2 = datamut2[[x for x in np.where(r2)]]
                            gs2.append(len(gi2))

                            #recreate normal distribution with 2 fitted GM from tail2
                            weightso2 = gmmmut4.weights_
                            meanso2 = gmmmut4.means_
                            covarso2 = gmmmut4.covariances_
                            ysso = weightso2[0]*stats.multivariate_normal.pdf(xs2, mean = meanso2[0][0],cov = covarso2[0][0])
                            ysso2 = weightso2[1]*stats.multivariate_normal.pdf(xs2, mean = meanso2[1][0],cov = covarso2[1][0])
                            expectedo2 = (ysso+ysso2)*dx2*np.sum(observed2)

                            #calculate chi-square
                            arro2= np.array([[x,y] for x,y in zip(observed2,expectedo2)])
                            c4 = sum([(x[0]-x[1])**2./x[1] for x in arro2])
                            p4 = 1-stats.chi2.cdf(c4,dof)
                            chiv4.append(c4),pv4.append(p4)

                            #reset xs
                            xs = np.linspace(np.min(data),np.max(data), num=499) #recreate normal distribution
                            xxs = np.linspace(np.min(data),np.max(data), num=500)
                            dx =np.diff(xxs) #calculate the integral value dx


                            if counting == 0:
                                #degrees of freedom factor
                                ctf = round(cc*tune_factor,2)
                                if x==1:
                                    chis.append(ctf)
                                    cclemon.append([ctf,ctf,ctf,ctf])
                                    biclemon.append([n_comp,n_comp])
                                    realbic.append([BIC1,BIC2])
                                    rem.append(1)
                            else:
                                #chisquare tunning factor
                                ctf = round(chis[-1]*tune_factor,2)
                            #stop when p value is lower than <0.05 , find what condition?
                            fc = 0

                            if n_comp1 == 1:
                                if x!= None:
                                    if c < ctf:
                                        if verbosity == True:
                                            print('Removed %d datapoints from the left tail and fitting with 1 Gaussian Mixture with Chi-square value %s <= %s' %(x,np.round(c,2),ctf))
                                        count=1
                                        fc = np.round(c,2)
                            if n_comp2 == 1:
                                if y!= None:
                                    if c3 < ctf:
                                        if verbosity == True:
                                            print('Removed %d datapoints from the right tail and fitting with 1 Gaussian Mixture with Chi-square value %s <= %s' %(y,np.round(c3,2),ctf))
                                        count=3
                                        fc = np.round(c3,2)
                            if n_comp1 == 2:
                                if x!=None:
                                    if len(gi1)<2:
                                        if c2 < ctf/2:
                                            if verbosity == True:
                                                print('Removed %d datapoints from the left tail and fitting with 2 Gaussian Mixture with Chi-square value %s <= %s' %(x,np.round(c2,2),ctf))
                                            count=2
                                            fc = np.round(c2,2)
                            if n_comp2 == 2:
                                if y!= None:
                                    if len(gi2)<2:
                                        if c4 < ctf/2:
                                            if verbosity == True:
                                                print('Removed %d datapoints from the right tail and fitting with 2 Gaussian Mixture with Chi-square value %s <= %s' %(y,np.round(c4,2),ctf))
                                            count=4 
                                            fc = np.round(c4,2)


                            cclemon.append([c,c2,c3,c4])
                            biclemon.append([n_comp1,n_comp2])
                            realbic.append([BIC1,BIC2])
                            if count >0: #only begin recording when it fits under one of the conditions above  
                                conditions.append(count)
                                chis.append(fc)
                                counting+=1
                            if count == 1 or count ==2:
                                rem.append(x)
                                BICs.append(BIC1)
                            elif count == 3 or count ==4:
                                rem.append(y)
                                BICs.append(BIC2)

                        if count ==0 :
                            count=5 #fifth condition where it did not fix
                            break

                    # -----------------------------------------------rerun GMM----------------------------------------------------              
                    condition = conditions[-1]
                    remove = rem[-1]

                    if condition == 1:
                        datamut = datas[remove:]
                        n_comp=1
                        gmm = GMM(n_components =1,tol=tol,random_state=41).fit(datamut)
                    elif condition == 2:
                        datamut = datas[remove:]
                        n_comp=2
                        gmm = GMM(n_components =2,tol=tol,random_state=41).fit(datamut)
                    elif condition == 3:
                        datamut = datas[:-remove]
                        n_comp=1
                        gmm = GMM(n_components =1,tol=tol,random_state=41).fit(datamut)
                    elif condition == 4:    
                        datamut = datas[:-remove]
                        n_comp=2
                        gmm = GMM(n_components =2,tol=tol,random_state=41).fit(datamut)
                    elif condition == 5:
                        n_comp=2
                        gmm = GMM(n_components =2,tol=tol,random_state=41).fit(data)
                        if verbosity == True:
                            print('Chi-square Method Did Not Fix the Tail Problem')

                    #BIC
                    if condition == 1 or condition ==2 or condition == 3 or condition == 4:
                        BIC = BICs[-1]
                    else:
                        n_components = np.arange(1,n_mod)
                        models = [GMM(n,tol=tol,random_state=41).fit(data)
                                  for n in n_components]

                        #BIC and AIC for validating optimal n
                        BIC = [m.bic(data) for m in models]

                        #optimal n_comp
                        n_comp = np.argmin(BIC)+1


                    # use guassian mixture modeling to model bimodal distribution
                    if condition == 1 or condition  ==2:
                        x_axis2 = np.sort(np.array([b for a in datamut for b in a]))
                    elif condition == 3 or condition ==4:
                        x_axis2 = np.sort(np.array([b for a in datamut for b in a]))
                    else:
                        x_axis2 = np.sort(np.array([b for a in data for b in a]))

                    #recreate normal distribution with fitted GM
                    means = gmm.means_
                    covars = gmm.covariances_
                    weights = gmm.weights_
                    if n_comp > 1:
                        yss = weights[0]*stats.multivariate_normal.pdf(xs, mean = means[0][0],cov = covars[0][0])
                        yss2 = weights[1]*stats.multivariate_normal.pdf(xs, mean = means[1][0],cov = covars[1][0])
                        expected = yss*dx*len(data)
                        expectedt = yss2*dx*len(data)
                        expectedboth = (yss+yss2)*dx*len(data)
                    else: 
                        yss = stats.multivariate_normal.pdf(xs, mean = means[0][0],cov = covars[0][0])
                        expected = yss*dx*len(data)

                    #finding out groups
                    groups = [np.argmax(a) for a in gmm.predict_proba(x_axis2.reshape(-1,1)).round(0)] #find which group each data point is under

                    #bimodality 

                    roll = groups != np.roll(groups,1)
                    roll[0] = False  #roll will make the first variable True but we do not want that
                    group_index = [x for x in np.where(roll)]
                    group_div = x_axis2[group_index]

        elif log2transform == False:
            #GMM parameters
            tol = 1e-8 #convergence threshold
            n_mod = 3 #number of models for BIC comparison

            #BIC
            n_components = np.arange(1,n_mod)
            models = [GMM(n,tol=tol,random_state=41).fit(datanorm) #state=41 to make sure it doesn't always start random
                      for n in n_components]

            #BIC and AIC for validating optimal n
            BIC = [m.bic(datanorm) for m in models]

            #use BIC as our standard due to limited datanorm 
            n_comp = np.argmin(BIC)+1

            # use guassian mixture modeling to model bimodal distribution
            gmm = GMM(n_components =n_comp,tol=tol,random_state=41).fit(datanorm)

            #sort for cutoff 
            x_axis2 = np.sort(np.array([b for a in datanorm for b in a]))

            #recreate logscaled distribution
            xs = np.linspace(np.min(datanorm),np.max(datanorm), num=499) #recreate normal distribution
            xxs = np.linspace(np.min(datanorm),np.max(datanorm), num=500)
            dx =np.diff(xxs) #calculate the integral value dx

            #recreate normal distribution with fitted GM
            means = gmm.means_
            covars = gmm.covariances_
            weights = gmm.weights_
            if n_comp > 1:
                yss = weights[0]*stats.multivariate_normal.pdf(xs, mean = means[0][0],cov = covars[0][0])
                yss2 = weights[1]*stats.multivariate_normal.pdf(xs, mean = means[1][0],cov = covars[1][0])
                expected = yss*dx*len(datanorm)
                expectedt = yss2*dx*len(datanorm)
                expectedboth = (yss+yss2)*dx*len(datanorm)
            else: 
                yss = stats.multivariate_normal.pdf(xs, mean = means[0][0],cov = covars[0][0])
                expected = yss*dx*len(datanorm)

            #finding out groups
            groups = [np.argmax(a) for a in gmm.predict_proba(x_axis2.reshape(-1,1)).round(0)] #find which group each datanorm point is under

            #bimodality 

            roll = groups != np.roll(groups,1)
            roll[0] = False  #roll will make the first variable True but we do not want that
            group_index = [x for x in np.where(roll)]
            group_div = x_axis2[group_index]

        #     ------------------------------------measure chi-square of fitted GMM-------------------------------------------------
        #     determin number of bin using dynamic binning
            datas = np.array([[a] for a in np.sort(np.array([b for a in datanorm for b in a]))]) 
            nums=int(1.88*len(datas)**(2/5)) #mann and wald 

            observed, bin_edges = np.histogram(datanorm, bins=np.linspace(np.min(datanorm),np.max(datanorm), num=nums), density=False)
            if dynamic_binning_s == True:
                golddof = len(observed)
                lenn = 0
                while lenn < golddof: #Loop until you have the ssame dof or unimprovable

                    observed, bin_edges,con = dynamic_binning(observed, bin_edges,final=False)

                    observed, what = np.histogram(datanorm, bins=bin_edges)

                    observed, bin_edges, con = dynamic_binning(observed, bin_edges)

                    observed, what = np.histogram(datanorm, bins=bin_edges)

                    if con == 0:
                        break

                    lenn = len(observed)

            #fit with a single GM
            xschi = bin_edges[1:] #recreate normal distribution using dynamically binned edges
            dxchi = np.diff(bin_edges) #calculate the integral value dx

            if n_comp > 1:
                ysschi = weights[0]*stats.multivariate_normal.pdf(xschi, mean = means[0][0],cov = covars[0][0])
                yss2chi = weights[1]*stats.multivariate_normal.pdf(xschi, mean = means[1][0],cov = covars[1][0])
                expectedchi = (ysschi+yss2chi)*dxchi*len(datanorm)
            else: 
                ysschi = stats.multivariate_normal.pdf(xschi, mean = means[0][0],cov = covars[0][0])
                expectedchi = ysschi*dxchi*len(datanorm)

            #calculate dof
            dof = len(observed)-1


            #calculate chi-square
            arrchi= np.array([[x,y] for x,y in zip(observed,expectedchi)])

            cc = sum([(x[0]-x[1])**2./x[1] for x in arrchi])
            pp = 1-stats.chi2.cdf(cc,dof)

            bimodalc = cc

            bins = 10 #used for histogram and chi-square
            count=0
            counting = 0 
            conditions = [5] #default with condition 5 meaning not fixed
            condition = 0 #default for those that doesn't go through any chi-square method
            chis = []
            rem = [0] #default with condition 5 meaning not fixed so removes nothing
            BICs = []
            cclemon = [] #record chisquare
            biclemon = [] #record BIC
            realbic = []
            nums=int(1.88*len(datanorm)**(2/5)) #mann and wald

        #     -----------------------------IF CHI-SQUARE VALUE > DOF WITH A SINGLE TAIL, RUN CHI-SQUARE METHOD-------
            if Single_tail_validation == True:
                if cc > dof and n_comp == 2 and len(group_div) < 2:
                    if max(weights)/min(weights) > 3: #IS THERE A TAIL????
                        if verbosity == True:
                            print('Single tail problem that may not be normally distributed, run chi-square method')
                        tail1 = [num for num in datanorm if num <= group_div]
                        tail2 = [num for num in datanorm if num >= group_div]

                        #deteremine which is the tail to begin trimming and fitting
                        if len(tail1) > len(tail2):
                            thetail = tail2
                            direct = 0
                        else:
                            thetail = tail1
                            direct = 1

                        #determine the number of max datanormpoints to take off
                        tailmax = math.ceil(len(thetail))
                        if tailmax < 0.2*len(datanorm):
                            tailmax = math.ceil(tailmax+farinto*tailmax)
                        if verbosity == True:
                            print('Number of datapoints in tail:', tailmax)     
                        datas = np.array([[a] for a in np.sort(np.array([b for a in datanorm for b in a]))]) #sort datanorm
                        if len(thetail) >0:

                            while True:
                                for x in range(1,tailmax):

                                    count=0 #re-zero counter to 0 every loop
                                    if direct == 0:
                                        datamut = datas[:-x]
                                    if direct == 1:
                                        datamut = datas[x:]

                                    nums=int(1.88*len(datamut)**(2/5)) #mann and wald

                                    #BIC
                                    models1 = [GMM(n,tol=tol,random_state=41).fit(datamut)
                                              for n in n_components]

                                    #BIC and AIC for validating optimal n
                                    BIC1 = [m.bic(datamut) for m in models1]

                                    #use BIC as our standard due to limited datanorm 
                                    n_comp1 = np.argmin(BIC1)+1

                                    #dynamically making sure we result in the same number of bins
                                    observed, bin_edges = np.histogram(datamut, 
                                                                       bins=np.linspace(np.min(datamut),np.max(datamut), num=nums), density=False)
                                    if dynamic_binning_s == True:
                                        golddof = len(observed)
                                        lenn = 0
                                        while lenn < golddof:

                                            observed, bin_edges,con = dynamic_binning(observed, bin_edges,final=False)

                                            observed, what = np.histogram(datamut, bins=bin_edges)

                                            observed, bin_edges,con = dynamic_binning(observed, bin_edges)

                                            observed, what = np.histogram(datamut, bins=bin_edges)

                                            if con == 0:
                                                break

                                            lenn = len(observed)


                                    #fit with a single GM
                                    gmmmut = GMM(n_components =1,tol=tol,random_state=41).fit(datamut)
                                    xs = bin_edges[1:] #recreate normal distribution using dynamically binned edges
                                    dx =np.diff(bin_edges) #calculate the integral value dx

                                    #recreate normal distribution with fitted GM
                                    means = gmmmut.means_
                                    covars = gmmmut.covariances_
                                    ys = stats.multivariate_normal.pdf(xs, mean = means[0][0],cov = covars[0][0])
                                    expected = dx*ys*np.sum(observed)


                                    #calculate dof
                                    dof = len(observed)-1

                                    #calculate chi-square
                                    arr= np.array([[x,y] for x,y in zip(observed,expected)])
                                    c = sum([(x[0]-x[1])**2./x[1] for x in arr])
                                    p = 1-stats.chi2.cdf(c,dof)

                                    #fit with two GMs
                                    gmmmut2 = GMM(n_components =2,tol=tol,random_state=41).fit(datamut)

                                    #figure where there are two groups still or not
                                    g1 = [np.argmax(a) for a in gmmmut2.predict_proba(datamut.reshape(-1,1)).round(0)] #find which group each datanorm point is under
                                    #bimodality 
                                    r1 = g1 != np.roll(g1,1)
                                    r1[0] = False  #roll will make the first variable True but we do not want that
                                    gi1 = datamut[[x for x in np.where(r1)]]

                                    #recreate normal distribution with 2 fitted GM
                                    weights2 = gmmmut2.weights_
                                    means2 = gmmmut2.means_
                                    covars2 = gmmmut2.covariances_

                                    yss = weights2[0]*stats.multivariate_normal.pdf(xs, mean = means2[0][0],cov = covars2[0][0])
                                    yss2 = weights2[1]*stats.multivariate_normal.pdf(xs, mean = means2[1][0],cov = covars2[1][0])
                                    expected2 = (yss+yss2)*dx*np.sum(observed)

                                    #calculate chi-square
                                    arr2= np.array([[x,y] for x,y in zip(observed,expected2)])
                                    c2 = sum([(x[0]-x[1])**2./x[1] for x in arr2])
                                    p2 = 1-stats.chi2.cdf(c2,dof)


                                    #reset xs
                                    xs = np.linspace(np.min(datanorm),np.max(datanorm), num=499) #recreate normal distribution
                                    xxs = np.linspace(np.min(datanorm),np.max(datanorm), num=500)
                                    dx =np.diff(xxs) #calculate the integral value dx

                                    #is it better than the original fit?
                                    if counting == 0:
                                        #degrees of freedom factor
                                        ctf = round(cc*tune_factor,2)
                                        if x == 1:
                                            cclemon.append([ctf,ctf])
                                            biclemon.append([n_comp])
                                            realbic.append(BIC1)
                                            chis.append(ctf)
                                            rem.append(1)
                                    else:
                                        #chisquare tunning factor
                                        ctf = round(chis[-1]*tune_factor,2)
                                    if n_comp1 == 1:
                                        if c < ctf:
                                            if verbosity == True:
                                                print('Removed %d datanormpoints and fitting with 1 Gaussian Mixture with Chi-square value %s <= %s' %(x,np.round(c,2),ctf))
                                            count=1
                                            fc = np.round(c,2)

                                    if n_comp1 == 2:
                                        if len(gi1)<2/2:
                                            if c2 <= ctf:
                                                if verbosity == True:
                                                    print('Removed %d datanormpoints and fitting with 2 Gaussian Mixture with Chi-square value %s <= %s' %(x,np.round(c2,2),ctf))
                                                count=2
                                                fc = np.round(c2,2)


                                    if count >0: #only begin recording when it fits under one of the conditions above  
                                        if direct == 1:
                                            conditions.append(count)
                                            chis.append(fc)
                                            counting+=1
                                            rem.append(x)
                                            BICs.append(BIC1)
                                        if direct == 0:
                                            conditions.append(count+2) #if direct == 0 means we are trimming from the right so add 2 for conditions and count
                                            chis.append(fc)
                                            counting+=1
                                            rem.append(x)
                                            BICs.append(BIC1)
                                            if n_comp1 ==1:
                                                count = 3
                                            if n_comp1 ==2:
                                                count = 4

                                    cclemon.append([c,c2])
                                    biclemon.append([n_comp1])
                                    realbic.append(BIC1)

                                if count ==0 :
                                    count=5 #fifth condition where it did not fix
                                    break

                            condition = conditions[-1]
                            remove = rem[-1]

                            if condition == 1:
                                datamut = datas[remove:]
                                n_comp=1
                                gmm = GMM(n_components =1,tol=tol,random_state=41).fit(datamut)
                            elif condition == 2:
                                datamut = datas[remove:]
                                n_comp=2
                                gmm = GMM(n_components =2,tol=tol,random_state=41).fit(datamut)
                            elif condition == 3:
                                datamut = datas[:-remove]
                                n_comp=1
                                gmm = GMM(n_components =1,tol=tol,random_state=41).fit(datamut)
                            elif condition == 4:    
                                datamut = datas[:-remove]
                                n_comp=2
                                gmm = GMM(n_components =2,tol=tol,random_state=41).fit(datamut)
                            elif condition == 5:
                                n_comp=2
                                gmm = GMM(n_components =2,tol=tol,random_state=41).fit(datanorm)
                                if verbosity == True:
                                    print('Chi-square Method Did Not Help the Tail Problem')

                            #BIC
                            if condition == 1 or condition ==2 or condition == 3 or condition == 4:
                                BIC = BICs[-1]
                            else:
                                n_components = np.arange(1,n_mod)
                                models = [GMM(n,tol=tol,random_state=41).fit(datanorm)
                                          for n in n_components]

                                #BIC and AIC for validating optimal n
                                BIC = [m.bic(datanorm) for m in models]

                                #optimal n_comp
                                n_comp = np.argmin(BIC)+1


                            # use guassian mixture modeling to model bimodal distribution
                            if condition == 1 or condition  ==2 or condition == 3 or condition ==4:
                                x_axis2 = np.sort(np.array([b for a in datamut for b in a]))
                            else:
                                x_axis2 = np.sort(np.array([b for a in datanorm for b in a]))

                            #recreate normal distribution with fitted GM
                            means = gmm.means_
                            covars = gmm.covariances_
                            weights = gmm.weights_

                            if n_comp > 1:
                                yss = weights[0]*stats.multivariate_normal.pdf(xs, mean = means[0][0],cov = covars[0][0])
                                yss2 = weights[1]*stats.multivariate_normal.pdf(xs, mean = means[1][0],cov = covars[1][0])
                                expected = yss*dx*len(datanorm)
                                expectedt = yss2*dx*len(datanorm)
                                expectedboth = (yss+yss2)*dx*len(datanorm)
                            else: 
                                yss = stats.multivariate_normal.pdf(xs, mean = means[0][0],cov = covars[0][0])
                                expected = yss*dx*len(datanorm)

                            #finding out groups
                            groups = [np.argmax(a) for a in gmm.predict_proba(x_axis2.reshape(-1,1)).round(0)] #find which group each datanorm point is under

                            #bimodality 

                            roll = groups != np.roll(groups,1)
                            roll[0] = False  #roll will make the first variable True but we do not want that
                            group_index = [x for x in np.where(roll)]
                            group_div = x_axis2[group_index]
                        else:
                            n_comp=1
        #     ----------------------------IF TAIL PROBELM RUN ------------------------------------------------------------
            if chisquaremethod == True:
                if len(group_div) > 1:
                    if verbosity == True:
                        print('Rerunning GMM with Chi-square Method to fix tail problem')

                    tail1 = [num for num in datanorm if num <= np.min(group_div)]
                    tail2 = [num for num in datanorm if num >= np.max(group_div)]

                    chiv=[]
                    pv=[]
                    chiv2=[]
                    pv2=[]
                    chiv3=[]
                    pv3=[]
                    chiv4=[]
                    pv4=[]
                    BICs=[]
                    gs = []
                    gs2 = []


                    #determine the number of max datanormpoints to take off
                    xup = math.ceil(len(tail1))
                    yup = math.ceil(len(tail2))

                    #add to the boundary if the tail is very small to allow more datanorm taken off for validation
                    if xup < 0.2*len(datanorm):
                        xup = math.ceil(xup+2+xup*farinto)
                    if yup < 0.2*len(datanorm):
                        yup = math.ceil(yup+2+yup*farinto)
                    if verbosity == True:
                        print('Number of datapoints in the right tail:', xup)
                        print('Number of datapoints in the left tail:', yup)
                    while True:
                        for x, y in itert.zip_longest(range(1,xup),range(1,yup)): 


                            datas = np.array([[a] for a in np.sort(np.array([b for a in datanorm for b in a]))]) #sort datanorm so we are taking off the right tail
                            count=0 #re-zero counter to 0 every loop
                            try:
                                datamut = datas[x:] #tail1 datanorm
                                datamut2 = datas[:-y] #tail2 datanorm
                            except:
                                pass

                            nums=int(1.88*len(datamut)**(2/5)) #mann and wald

                            #BIC
                            models1 = [GMM(n,tol=tol,random_state=41).fit(datamut)
                                      for n in n_components]

                            #BIC and AIC for validating optimal n
                            BIC1 = [m.bic(datamut) for m in models1]

                            #use BIC as our standard due to limited datanorm 
                            n_comp1 = np.argmin(BIC1)+1

                        # ---------------------------------tail1----------------------------------------------------

                            observed, bin_edges = np.histogram(datamut, 
                                                               bins=np.linspace(np.min(datamut),np.max(datamut), num=nums), density=False)
                            if dynamic_binning_s == True:
                                golddof = len(observed)
                                lenn=0
                                while lenn < golddof:
                                    observed, bin_edges,con = dynamic_binning(observed, bin_edges,final=False)

                                    observed, what = np.histogram(datamut, bins=bin_edges)

                                    observed, bin_edges,con = dynamic_binning(observed, bin_edges)

                                    observed, what = np.histogram(datamut, bins=bin_edges)

                                    if con == 0:
                                        break

                                    lenn=len(observed)
                            #fit with a single GM
                            gmmmut = GMM(n_components =1,tol=tol,random_state=41).fit(datamut)
                            xs = bin_edges[1:] #recreate normal distribution using dynamically binned edges
                            dx =np.diff(bin_edges) #calculate the integral value dx

                            #recreate normal distribution with fitted GM
                            means = gmmmut.means_
                            covars = gmmmut.covariances_
                            ys = stats.multivariate_normal.pdf(xs, mean = means[0][0],cov = covars[0][0])
                            expected = dx*ys*np.sum(observed)


                            #calculate dof
                            dof = len(observed)-1

                            #calculate chi-square
                            arr= np.array([[x,y] for x,y in zip(observed,expected)])
                            c = sum([(x[0]-x[1])**2./x[1] for x in arr])
                            p = 1-stats.chi2.cdf(c,dof)
                            chiv.append(c),pv.append(p)

                            #fit with two GMs
                            gmmmut2 = GMM(n_components =2,tol=tol,random_state=41).fit(datamut)

                            #figure where there are two groups still or not
                            g1 = [np.argmax(a) for a in gmmmut2.predict_proba(datamut.reshape(-1,1)).round(0)] #find which group each datanorm point is under
                            #bimodality 
                            r1 = g1 != np.roll(g1,1)
                            r1[0] = False  #roll will make the first variable True but we do not want that
                            gi1 = datamut[[x for x in np.where(r1)]]
                            gs.append(len(gi1))

                            #recreate normal distribution with 2 fitted GM
                            weights2 = gmmmut2.weights_
                            means2 = gmmmut2.means_
                            covars2 = gmmmut2.covariances_
                            yss = weights2[0]*stats.multivariate_normal.pdf(xs, mean = means2[0][0],cov = covars2[0][0])
                            yss2 = weights2[1]*stats.multivariate_normal.pdf(xs, mean = means2[1][0],cov = covars2[1][0])
                            expected2 = (yss+yss2)*dx*np.sum(observed)

                            #calculate chi-square
                            arr2= np.array([[x,y] for x,y in zip(observed,expected2)])
                            c2 = sum([(x[0]-x[1])**2./x[1] for x in arr2])
                            p2 = 1-stats.chi2.cdf(c2,dof)
                            chiv2.append(c2),pv2.append(p2)

                            #BIC
                            models2 = [GMM(n,tol=tol,random_state=41).fit(datamut2)
                                      for n in n_components]

                            #BIC and AIC for validating optimal n
                            BIC2 = [m.bic(datamut2) for m in models2]

                            #use BIC as our standard due to limited datanorm 
                            n_comp2 = np.argmin(BIC2)+1
                        # ---------------------------------tail2----------------------------------------------------
                            observed2, bin_edges2 = np.histogram(datamut2, 
                                                               bins=np.linspace(np.min(datamut2),np.max(datamut2), num=nums), density=False)

                            if dynamic_binning_s == True:
                                golddof = len(observed2)
                                lenn=0
                                while len(observed2) < golddof:
                                    observed2, bin_edges2,con = dynamic_binning(observed2, bin_edges2,final=False)

                                    observed2, what = np.histogram(datamut2, bins=bin_edges2)

                                    observed2, bin_edges2,con = dynamic_binning(observed2, bin_edges2)

                                    observed2, what = np.histogram(datamut2, bins=bin_edges2)

                                    if con ==0:
                                        lenn=len(observed2)

                            #fit with a single GM
                            gmmmut3 = GMM(n_components =1,tol=tol,random_state=41).fit(datamut2)
                            xs2 = bin_edges2[1:] #recreate normal distribution
                            dx2 =np.diff(bin_edges2) #calculate the integral value dx

                            #recreate normal distribution with fitted GM cutting from tail2
                            meanso = gmmmut3.means_
                            covarso = gmmmut3.covariances_
                            yso = stats.multivariate_normal.pdf(xs2, mean = meanso[0][0],cov = covarso[0][0])
                            expectedo = dx2*yso*np.sum(observed2)

                            #calculate chi-square
                            arro= np.array([[x,y] for x,y in zip(observed2,expectedo)])
                            c3 = sum([(x[0]-x[1])**2./x[1] for x in arro])
                            p3 = 1-stats.chi2.cdf(c3,dof)
                            chiv3.append(c3),pv3.append(p3)
                            #fit with two GMs
                            gmmmut4 = GMM(n_components =2,tol=tol,random_state=41).fit(datamut2)

                            #figure where there are two groups still or not
                            g2 = [np.argmax(a) for a in gmmmut4.predict_proba(datamut2.reshape(-1,1)).round(0)] #find which group each datanorm point is under
                            #bimodality 
                            r2 = g2!=np.roll(g2,1)
                            r2[0] = False  #roll will make the first variable True but we do not want that
                            gi2 = datamut2[[x for x in np.where(r2)]]
                            gs2.append(len(gi2))

                            #recreate normal distribution with 2 fitted GM from tail2
                            weightso2 = gmmmut4.weights_
                            meanso2 = gmmmut4.means_
                            covarso2 = gmmmut4.covariances_
                            ysso = weightso2[0]*stats.multivariate_normal.pdf(xs2, mean = meanso2[0][0],cov = covarso2[0][0])
                            ysso2 = weightso2[1]*stats.multivariate_normal.pdf(xs2, mean = meanso2[1][0],cov = covarso2[1][0])
                            expectedo2 = (ysso+ysso2)*dx2*np.sum(observed2)

                            #calculate chi-square
                            arro2= np.array([[x,y] for x,y in zip(observed2,expectedo2)])
                            c4 = sum([(x[0]-x[1])**2./x[1] for x in arro2])
                            p4 = 1-stats.chi2.cdf(c4,dof)
                            chiv4.append(c4),pv4.append(p4)

                            #reset xs
                            xs = np.linspace(np.min(datanorm),np.max(datanorm), num=499) #recreate normal distribution
                            xxs = np.linspace(np.min(datanorm),np.max(datanorm), num=500)
                            dx =np.diff(xxs) #calculate the integral value dx


                            if counting == 0:
                                #degrees of freedom factor
                                ctf = round(cc*tune_factor,2)
                                if x==1:
                                    chis.append(ctf)
                                    cclemon.append([ctf,ctf,ctf,ctf])
                                    biclemon.append([n_comp,n_comp])
                                    realbic.append([BIC1,BIC2])
                                    rem.append(1)
                            else:
                                #chisquare tunning factor
                                ctf = round(chis[-1]*tune_factor,2)
                            #stop when p value is lower than <0.05 , find what condition?
                            fc = 0
                            if n_comp1 == 1:
                                if x!= None:
                                    if c < ctf:
                                        if verbosity == True:
                                            print('Removed %d datanormpoints from the left tail and fitting with 1 Gaussian Mixture with Chi-square value %s <= %s' %(x,np.round(c,2),ctf))
                                        count=1
                                        fc = np.round(c,2)
                            if n_comp2 == 1:
                                if y!= None:
                                    if c3 < ctf:
                                        if verbosity == True:
                                            print('Removed %d datanormpoints from the right tail and fitting with 1 Gaussian Mixture with Chi-square value %s <= %s' %(y,np.round(c3,2),ctf))
                                        count=3
                                        fc = np.round(c3,2)
                            if n_comp1 == 2:
                                if x!=None:
                                    if len(gi1)<2:
                                        if c2 < ctf/2:
                                            if verbosity == True:
                                                print('Removed %d datanormpoints from the left tail and fitting with 2 Gaussian Mixture with Chi-square value %s <= %s' %(x,np.round(c2,2),ctf))
                                            count=2
                                            fc = np.round(c2,2)
                            if n_comp2 == 2:
                                if y!= None:
                                    if len(gi2)<2:
                                        if c4 < ctf/2:                                    
                                            if verbosity == True:
                                                print('Removed %d datanormpoints from the right tail and fitting with 2 Gaussian Mixture with Chi-square value %s <= %s' %(y,np.round(c4,2),ctf))
                                            count=4 
                                            fc = np.round(c4,2)


                            cclemon.append([c,c2,c3,c4])
                            biclemon.append([n_comp1,n_comp2])
                            realbic.append([BIC1,BIC2])
                            if count >0: #only begin recording when it fits under one of the conditions above  
                                conditions.append(count)
                                chis.append(fc)
                                counting+=1
                            if count == 1 or count ==2:
                                rem.append(x)
                                BICs.append(BIC1)
                            elif count == 3 or count ==4:
                                rem.append(y)
                                BICs.append(BIC2)

                        if count ==0 :
                            count=5 #fifth condition where it did not fix
                            break

                    # -----------------------------------------------rerun GMM----------------------------------------------------              
                    condition = conditions[-1]
                    remove = rem[-1]

                    if condition == 1:
                        datamut = datas[remove:]
                        n_comp=1
                        gmm = GMM(n_components =1,tol=tol,random_state=41).fit(datamut)
                    elif condition == 2:
                        datamut = datas[remove:]
                        n_comp=2
                        gmm = GMM(n_components =2,tol=tol,random_state=41).fit(datamut)
                    elif condition == 3:
                        datamut = datas[:-remove]
                        n_comp=1
                        gmm = GMM(n_components =1,tol=tol,random_state=41).fit(datamut)
                    elif condition == 4:    
                        datamut = datas[:-remove]
                        n_comp=2
                        gmm = GMM(n_components =2,tol=tol,random_state=41).fit(datamut)
                    elif condition == 5:
                        n_comp=2
                        gmm = GMM(n_components =2,tol=tol,random_state=41).fit(datanorm)
                        if verbosity == True:
                            print('Chi-square Method Did Not Fix the Tail Problem')

                    #BIC
                    if condition == 1 or condition ==2 or condition == 3 or condition == 4:
                        BIC = BICs[-1]
                    else:
                        n_components = np.arange(1,n_mod)
                        models = [GMM(n,tol=tol,random_state=41).fit(datanorm)
                                  for n in n_components]

                        #BIC and AIC for validating optimal n
                        BIC = [m.bic(datanorm) for m in models]

                        #optimal n_comp
                        n_comp = np.argmin(BIC)+1


                    # use guassian mixture modeling to model bimodal distribution
                    if condition == 1 or condition  ==2:
                        x_axis2 = np.sort(np.array([b for a in datamut for b in a]))
                    elif condition == 3 or condition ==4:
                        x_axis2 = np.sort(np.array([b for a in datamut for b in a]))
                    else:
                        x_axis2 = np.sort(np.array([b for a in datanorm for b in a]))

                    #recreate normal distribution with fitted GM
                    means = gmm.means_
                    covars = gmm.covariances_
                    weights = gmm.weights_
                    if n_comp > 1:
                        yss = weights[0]*stats.multivariate_normal.pdf(xs, mean = means[0][0],cov = covars[0][0])
                        yss2 = weights[1]*stats.multivariate_normal.pdf(xs, mean = means[1][0],cov = covars[1][0])
                        expected = yss*dx*len(datanorm)
                        expectedt = yss2*dx*len(datanorm)
                        expectedboth = (yss+yss2)*dx*len(datanorm)
                    else: 
                        yss = stats.multivariate_normal.pdf(xs, mean = means[0][0],cov = covars[0][0])
                        expected = yss*dx*len(datanorm)

                    #finding out groups
                    groups = [np.argmax(a) for a in gmm.predict_proba(x_axis2.reshape(-1,1)).round(0)] #find which group each datanorm point is under

                    #bimodality 

                    roll = groups != np.roll(groups,1)
                    roll[0] = False  #roll will make the first variable True but we do not want that
                    group_index = [x for x in np.where(roll)]
                    group_div = x_axis2[group_index]

        if log2transform == True:
            # -------------------------------------------PLOTTTTTTTTTTTTTTTTT------------------------------------------
            x_axis = np.sort(np.array([b for a in datanorm[np.nonzero(datanorm)].reshape(-1,1) for b in a])) #reset x_axis for plotting

            if graphs == True:
                f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=[18,14.5])

                axisfont = 18
                titlefont = 18.5

                a2 = ax1.twinx()
                #plot 
            #         a2.set_xscale('log',basex=2)
                if n_comp > 1:
                    a2.plot(2**xs,expected,'--k', color='r',alpha=.5)
                    a2.plot(2**xs,expectedt,'--k', color='r',alpha=.5)
                    a2.plot(2**xs,expectedboth, '-k', color='b',alpha=.5)
                else:
                    a2.plot(2**xs,expected,'-k', color='b',alpha=.5)


                a2.set_ylim(bottom=0)
                a2.set_yticks([])

                #plot histogram of original data
                ax1.set_title('Log2 Transform Data GMM Fit of :' + ' '+ ID, fontsize=titlefont)

            #     use this for frequency instead of counts
            #     n, bins, patches = ax1.hist(datanorm,weights=np.ones(len(datanorm))/len(datanorm), 
            #              bins = np.logspace(np.log2(np.min(x_axis)),np.log2(np.max(x_axis)), num=nums,base=2),histtype='barstacked',alpha=0.5, ec='black')

                n, bins, patches = ax1.hist(datanorm, 
                         bins = np.logspace(np.log2(np.min(x_axis)),np.log2(np.max(x_axis)),num=nums,base=2),histtype='barstacked',alpha=0.5, ec='black')    
                ax1.set_ylabel('Number of Samples',fontsize=axisfont)
                ax1.set_xlabel('Expression Value (log2)',fontsize=axisfont)
                ax1.set_xscale('log',basex = 2)

                logfmt = tick.LogFormatterExponent(base=2.0, labelOnlyBase=True) #take out the 2^ just have the powers
                ax1.xaxis.set_major_formatter(logfmt)
                ax1.tick_params(axis='both',labelsize=axisfont)
                a2.tick_params(axis='both',labelsize=axisfont)
                ax1.grid('on')

                twostd = meanf+2*stdf
                if calc_back == False and calc_backpara == True:
                    #color potential background, x < threshold
                    if sum(data < 2**twostd) >1 :
                        ind = [i for i, y in enumerate(bins) if y <= 2**(meanf+2*stdf)]
                        if len(ind)>1:
                            for i in range(ind[0],ind[-1]):
                                patches[i].set_facecolor('orange')
                    if np.min(data) < 2**filt:
                        ind = [i for i, y in enumerate(bins) if y <= 2**filt]
                        if len(ind)>1:
                            for i in range(ind[0],ind[-1]):
                                patches[i].set_facecolor('r')

                    background_p = mpatches.Patch(facecolor='lightcoral',ec = 'black',label='Background Threshold')
                    background_p2 = mpatches.Patch(facecolor='gold',ec = 'black',label='Two \u03C3 Background Threshold')

                    a2.legend(handles=[background_p, background_p2])

                if find_target == True:
                    n2, bins2, patches2 = ax1.hist(input_datanorm[cell_lines].loc[ID],hatch='///',facecolor='b',
                                                   bins = np.logspace(np.log2(np.min(x_axis)),np.log2(np.max(x_axis)),num=nums,base=2),histtype='barstacked',alpha=0.5, ec='black') 
                    highlight = input_datanorm[cell_lines].loc[ID]
                    ind = []
                    highlight_p = mpatches.Patch(hatch='///', facecolor = 'w',edgecolor='black',label='Sample(s) of Interest') #add to legend
                    a2.legend(handles=[background_p, background_p2,highlight_p])

                    n2, bins2, patches2 = ax1.hist(input_datanorm[cell_lines].loc[ID],hatch='///',facecolor='b',
                                                   bins = np.logspace(np.log2(np.min(x_axis)),np.log2(np.max(x_axis)),num=nums,base=2),
                                                   histtype='barstacked',alpha=0.5, ec='black') 
                    #color in with threshold
                    if sum(data < 2**twostd) >1 :
                        ind = [i for i, y in enumerate(bins2) if y <= 2**(meanf+2*stdf)]
                        if len(ind)>1:
                            for i in range(ind[0],ind[-1]):
                                patches2[i].set_facecolor('orange')
                    if np.min(data) < 2**filt:
                        ind = [i for i, y in enumerate(bins2) if y <= 2**filt]
                        if len(ind)>1:
                            for i in range(ind[0],ind[-1]):
                                patches2[i].set_facecolor('r')

                    i = 0 #python zero index
                    ind=[] #zero it
                    for x, y in zip(bins[:-1], np.roll(bins,-1)[:-1]): #find if target value is between bins
                        for z in highlight:
                            if x<=z<=y:
                                ind.append(i)
                        i+=1

                    #show number in that bin
                    i = 0 #python zero index
                    values = pd.value_counts(ind).index
                    counted = list(pd.value_counts(ind))
                    for i, v in enumerate(counted):
                        x = bins[:-1]
                        y = np.roll(bins,-1)[:-1]
                        ax1.text((x[values[i]]+y[values[i]])/2, n[values[i]] + max(n)/50, '(%s)'%v,
                                 horizontalalignment='center', fontsize=axisfont)


                #deem the output unimodal or bimodal whether chi-square method was applied
                intersections = 2**group_div

                if count == 0:
                    if n_comp > 1:
                        for a in intersections: ax1.axvline(a, linestyle='--', c='g')
                        print('Bimodal, Cutoff Threshold:', group_div)
                        xf = group_div[0]
                        classif = 2
                    else:
                        pass
                        print('Unimodal')
                        xf = None
                        classif = 1
                elif condition == 1: #remove 1 point means removeing 0 index need to zero index
                    ax1.axvline(2**datas[remove][0], linestyle='--', c='g')
                    print('Removed %s datapoints from the left tail, Two Categories by Chi-square Method, Cutoff Threshold: %s' %(remove,datas[remove][0]))
                    xf = datas[remove][0]
                    classif = 21
                elif condition == 2:
                    ax1.axvline(2**datas[remove][0], linestyle='--', c='g')
                    for a in intersections: ax1.axvline(a, linestyle='--', c='g')
                    print('Removed %s datapoints from the left tail, Three Categories by Chi-square Method, Cutoff Threshold: %s' %(remove,np.sort(np.concatenate([group_div,datas[remove]]))))
                    xf = np.sort(np.concatenate([group_div,datas[remove]]))
                    classif = 3
                elif condition == 3:
                    remd = remove+1 #not zero indexed coming from the other side
                    ax1.axvline(2**datas[-remd][0], linestyle='--', c='g')
                    print('Removed %s datapoints from the right tail, Two Categories by Chi-square Method, Cutoff Threshold: %s'% (remove,datas[-remove][0]))
                    xf = datas[-remove][0]
                    classif = 21
                elif condition == 4:
                    remd = remove+1 
                    ax1.axvline(2**datas[-remd][0], linestyle='--', c='g')
                    for a in intersections: ax1.axvline(a, linestyle='--', c='g')
                    print('Removed %s datapoints from the right tail, Three Categories by Chi-square Method, Cutoff Threshold: %s'%(remove,np.sort(np.concatenate([group_div,datas[-remove]]))))
                    xf = np.sort(np.concatenate([group_div,datas[-remove]]))
                    classif = 3
                elif condition == 5:
                    for a in intersections: ax1.axvline(a, linestyle='--', c='g')
                    print('Bimodal cannot be helped by Chi-square Method, Cutoff Threshold:', group_div)
                    xf = group_div[0]
                    classif = 20
            # ----------------------------------------------------------------------------------------------------------------------
                #BIC and AIC graph
                ax2.set_title('BIC Graph of :' + ' '+ ID, fontsize=titlefont)
                ax2.tick_params(axis='both',labelsize=axisfont)

                if condition != 0 and condition !=5:
                    xp=np.arange(0,len(biclemon))
                    length = len(biclemon[0])
                    ax2.set_ylim((0.8,2.2))
                    ax2.set_yticks([1,2])
                    ax2.set_ylabel('n_components',fontsize=axisfont)
                    ax2.set_xlabel('Datapoints Removed',fontsize=axisfont);

                    ax22 = ax2.twinx() #plot BIC value
                    ax22.tick_params(axis='both',labelsize=axisfont)

                    if  length == 1:
                        ax2.plot(xp,biclemon,'--',c ='grey')

                        color = ['b','r']
                        for zz in range(0,len(realbic[0])):
                            bicbag = []
                            for bics in realbic:
                                bicbag.append(bics[zz])
                            ax22.plot(xp,bicbag,'-',c = color[zz])

                        b1 = mlines.Line2D([], [], color=color[0], label='1 n_comp w/ BIC')
                        b2 = mlines.Line2D([], [], color=color[1], label='2 n_comp w/ BIC')
                        b3 = mlines.Line2D([], [], color='grey', label='n_comp w/ Lower BIC Score')
                        ax22.legend(handles=[b1,b2,b3])
                        ax22.set_ylabel('BIC Score',fontsize=axisfont)
                    else:
                        color = ['grey','black']
                        for zz in range(0,length):
                            bicbags = []
                            for bics in biclemon:
                                bicbags.append(bics[zz])
                            ax2.plot(xp,bicbags,'--',c = color[zz])

                        colors = ['deepskyblue','dodgerblue','tomato','lightcoral']

                        newbic = [np.concatenate(x) for x in realbic]
                        for zz in range(0,len(newbic[0])):
                            bicbag = []
                            for bics in newbic:
                                bicbag.append(bics[zz])
                            ax22.plot(xp,bicbag,'-',c = colors[zz])

                        bb1 = mlines.Line2D([], [], color=colors[0], label='Left Tail w/ 1 n_comp BIC')
                        bb2 = mlines.Line2D([], [], color=colors[1], label='Left Tail w/ 2 n_comp BIC')
                        bb3 = mlines.Line2D([], [], color=colors[2], label='Right Tail w/ 1 n_comp BIC')
                        bb4 = mlines.Line2D([], [], color=colors[3], label='Right Tail w/ 2 n_comp BIC')  
                        b1 = mlines.Line2D([], [], color=color[0], label='Left tail n_comp w/ Lower BIC')
                        b2 = mlines.Line2D([], [], color=color[1], label='Right tail n_comp w/ Lower BIC')
                        ax22.legend(handles=[bb1,bb2,bb3,bb4,b1,b2])
                        ax22.set_ylabel('BIC Score',fontsize=axisfont)
                else:
                    ax2.bar(n_components, BIC, width = 0.2, label='BIC',align='center',ec='b')
                    ax2.set_xticks([1,2])
                    ax2.set_ylim(bottom = np.min(BIC)-5, top = np.max(BIC)+2)
                    ax2.set_ylabel('BIC Score',fontsize=axisfont)
                    ax2.set_xlabel('n_components',fontsize=axisfont);

            # ----------------------------------------------------------------------------------------------------------------------
                #plot number of samples in each category

                ax3.grid('on')
                ax3.tick_params(axis='both',labelsize=axisfont)
                cutoff = xf
                try:
                    if len(cutoff) == 1:
                        cutoff = cutoff[0]  
                except:
                    pass

                if sum(cutoff == None)==1:
                    ax3.text(0.35,0.5,'Unimodal', fontsize=titlefont)
                    categories=np.ones(len(data)).tolist()
                elif isinstance(cutoff,float) | isinstance(cutoff,np.integer):

                    if calc_back == False:
                        #determine patients
                        low_exp_group = input_data.columns[(input_data<cutoff).loc[ID]]
                        high_exp_group = input_data.columns[(input_data>=cutoff).loc[ID]]
                    else:
                        #determine datapoints
                        low_exp_group = data[data <= cutoff]
                        high_exp_group = data[data > cutoff]


                    tol = len(low_exp_group) +len(high_exp_group)
                    print(' No. in Low Expression Group: %s (%s%%)' %(len(low_exp_group),np.round((len(low_exp_group)/tol)*100,2)),'\n', 
                      'No. in High Expression Group: %s (%s%%)' %(len(high_exp_group), np.round((len(high_exp_group)/tol)*100,2)),'\n',
                      'Number of Total:', tol)

                    y = np.round([len(low_exp_group),len(high_exp_group)],2)
                    y_p = np.round([len(low_exp_group)*100/tol,len(high_exp_group)*100/tol],2)
                    x = np.arange(2)

                    ax3.bar(x,y,width=0.2,align='center',ec='b')
                    ax3.set_xticks(x)
                    ax3.set_xticklabels(['Low','High'],fontsize=axisfont)
                    ax3.set_title('Expression Categories of %s'%ID,fontsize=titlefont)
                    ax3.set_ylabel('Number of Samples',fontsize=axisfont)

                    if calc_back == False:
                        #determine datapoints
                        x_lowexp = pd.DataFrame(input_data.loc[:,low_exp_group])
                        x_highexp = pd.DataFrame(input_data.loc[:,high_exp_group])




                        if calc_backpara == True:
                            true_posh = x_highexp.columns[(x_highexp>twostd).loc[ID]]


                            categories = [2 if x in true_posh else 1 for x in input_datanormcat]

                            y2 = np.round([len([]),len(true_posh)],2)

                            ax3.bar(x,y2,width=0.2,align='center',ec='b',facecolor='g')

                            background_d = mpatches.Patch(facecolor='green',ec = 'black',
                                                          label='True Positive(s): %s samples'%(len(true_posh)))
                            ax3.legend(handles=[background_d])
                        else:
                            categories = [2 if x in x_highexp.columns else 1 for x in input_datanormcat]





                    for i, v in enumerate(y_p):
                        ax3.text(x[i]-0.05, y[i] + max(y)/100, str(v)+'%', fontsize=axisfont)

                elif len(cutoff) ==2:
                    if calc_back == False:
                        #determine patients
                        low_exp_group = input_data.columns[(input_data<cutoff[0]).loc[ID]]
                        med_exp_group = input_data.columns[(cutoff[0] <= input_data.loc[ID]) & (input_data.loc[ID] < cutoff[1])]
                        high_exp_group = input_data.columns[(input_data>=cutoff[1]).loc[ID]]
                    else:
                        #determine datapoints
                        low_exp_group = data[data < cutoff[0]]
                        med_exp_group = data[(cutoff[0] <= data) & (data < cutoff[1])]
                        high_exp_group = data[data >= cutoff[1]]

                    tol = len(low_exp_group) +len(high_exp_group)+len(med_exp_group)
                    print(' No. in Low Expression Group: %s (%s%%)' %(len(low_exp_group),np.round((len(low_exp_group)/tol)*100,2)),'\n',
                          'No. in Med Expression Group: %s (%s%%)' %(len(med_exp_group),np.round((len(med_exp_group)/tol)*100,2)),'\n',
                      'No. in High Expression Group: %s (%s%%)' %(len(high_exp_group), np.round((len(high_exp_group)/tol)*100,2)),'\n',
                      'Number of Total:', tol)

                    y = np.round([len(low_exp_group),len(med_exp_group),len(high_exp_group)],2)
                    y_p = np.round([len(low_exp_group)*100/tol,len(med_exp_group)*100/tol,len(high_exp_group)*100/tol],2)
                    x = np.arange(3)


                    ax3.bar(x,y,width=0.2,align='center',ec='b')
                    ax3.set_xticks(x)
                    ax3.set_xticklabels(['Low','Med','High'],fontsize=axisfont)
                    ax3.set_title('Expression Categories of %s'%ID,fontsize=titlefont)
                    ax3.set_ylabel('Number of Samples', fontsize=axisfont)

                    if calc_back == False:

                        #determine datapoints
                        x_lowexp = pd.DataFrame(input_data.loc[:,low_exp_group])
                        x_medexp = pd.DataFrame(input_data.loc[:,med_exp_group])
                        x_highexp = pd.DataFrame(input_data.loc[:,high_exp_group])



                        if calc_backpara == True:
                            true_posh = x_highexp.columns[((x_highexp>twostd).loc[ID])]
                            true_posm = x_medexp.columns[((x_medexp>twostd).loc[ID])]
                            categories = [2 if (x in true_posh) | (x in true_posm) else 1 for x in input_datanormcat]

                            y2 = np.round([len([]),len(true_posm),len(true_posh)],2)

                            ax3.bar(x,y2,width=0.2,align='center',ec='b',facecolor='g')

                            background_d = mpatches.Patch(facecolor='green',ec = 'black',
                                                          label='True Positive(s) = %s samples'%(len(true_posh)+len(true_posm)))
                            ax3.legend(handles=[background_d])

                        else:
                            categories = [2 if (x in x_highexp.columns) | (x in x_medexp.columns) else 1 for x in input_datanormcat]


                    for i, v in enumerate(y_p):
                        ax3.text(x[i]-0.08, y[i]+max(y)/100, str(v)+'%',fontsize=axisfont)
            # ----------------------------------------------------------------------------------------------------------------------
                cclemon = np.array(cclemon)

                #plot chisquare value
                ax4.tick_params(axis='both',labelsize=axisfont)
                from mpl_toolkits.axes_grid1 import make_axes_locatable   

                if condition != 0 and condition !=5:
                    #chisquare value graph
                    xp = np.arange(0,len(cclemon))
                    length = len(cclemon[0])
                    if length == 2:

                        color = ['r','b']

                        l1 = mlines.Line2D([], [], color=color[0], label='Fitting w/ 1 n_comp')
                        l2 = mlines.Line2D([], [], color=color[1], label='Fitting w/ 2 n_comp')
                        dofnum = mpatches.Patch(color='none', label='Degrees of Freedom: %s'%dof)
                        arrows = mpatches.FancyArrowPatch(0.1,0.2,color = 'r' ,label = 'Lower X\N{SUPERSCRIPT TWO} Found' )

                        if np.max(cclemon[np.isfinite(cclemon)])/chis[0] >2.5: #if the max is not at least twice the initial then dont break axis
                            divider = make_axes_locatable(ax4)
                            ax44 = divider.new_vertical(size="100%", pad=0.1) #break the axis to show lower chi-square
                            f.add_axes(ax44)

                            ax4.set_ylim([0, chis[0]*1.1])
                            ax4.spines['top'].set_visible(False)
                            ax44.set_ylim([np.mean(cclemon[np.isfinite(cclemon)]), max(cclemon[np.isfinite(cclemon)])])
                            ax44.tick_params(bottom="off", labelbottom='off')
                            ax44.spines['bottom'].set_visible(False)
                            ax44.legend(handles=[l1,l2,arrows,dofnum])
                        else:
                            ax4.legend(handles=[l1,l2,arrows,dofnum])

                        for zz in range(0,length):
                            lemonbags=[]
                            for lemons in cclemon:
                                lemonbags.append(lemons[zz])
                            ax4.plot(xp,lemonbags,'-',c=color[zz])
                            if np.max(cclemon[np.isfinite(cclemon)])/chis[0] >2.5: #if the max is not at least twice the initial then dont break axis
                                ax44.plot(xp,lemonbags,'-',c=color[zz])

                        highlight = [[x,y] for x,y in zip(rem[2:],chis[1:])]

                        for h in highlight:
                            ax4.annotate('(%s)'%h[0], ha='center',fontsize = axisfont, xy=(h[0], h[1]), xytext=(0, 25), textcoords='offset points', 
                                        arrowprops=dict(ec='r',lw= 4))



                    else:            
                        highlight = [[x,y] for x,y in zip(rem[2:],chis[1:])]
                        color = ['r','b','g','orange']
                        l1 = mlines.Line2D([], [], color=color[0], label='Left Tail w/ 1 n_comp')
                        l2 = mlines.Line2D([], [], color=color[1], label='Left Tail w/ 2 n_comp')
                        r1 = mlines.Line2D([], [], color=color[2], label='Right Tail w/ 1 n_comp')
                        r2 = mlines.Line2D([], [], color=color[3], label='Right Tail w/ 2 n_comp')
                        arrows = mpatches.FancyArrowPatch(0.1,0.2,color = 'r' ,label = 'Lower X\N{SUPERSCRIPT TWO} Found' )
                        dofnum = mpatches.Patch(color='none', label='Degrees of Freedom: %s'%dof)

                        if np.max(cclemon[np.isfinite(cclemon)])/chis[0] >2.5: #if the max is not at least twice the initial then dont break axis
                            divider = make_axes_locatable(ax4)
                            ax44 = divider.new_vertical(size="100%", pad=0.1) #break the axis to show lower chi-square
                            f.add_axes(ax44)

                            ax4.set_ylim([0, chis[0]*1.1])
                            ax4.spines['top'].set_visible(False)
                            ax44.set_ylim([np.mean(cclemon[np.isfinite(cclemon)]),max(cclemon[np.isfinite(cclemon)])])
                            ax44.tick_params(bottom="off", labelbottom='off')
                            ax44.spines['bottom'].set_visible(False)
                            ax44.legend(handles=[l1,l2,r1,r2,arrows,dofnum])
                        else:
                            ax4.legend(handles=[l1,l2,r1,r2,arrows,dofnum])

                        for zz in range(0,length):
                            lemonbags=[]
                            for lemons in cclemon:
                                lemonbags.append(lemons[zz])
                            ax4.plot(xp,lemonbags,'-',c=color[zz])
                            if np.max(cclemon[np.isfinite(cclemon)])/chis[0] >2.5: #if the max is not at least twice the initial then dont break axis
                                ax44.plot(xp,lemonbags,'-',c=color[zz])

                        for h in highlight:
                            ax4.annotate('(%s)'%h[0], ha='center',fontsize = axisfont, xy=(h[0], h[1]), xytext=(0, 25), textcoords='offset points', 
                                        arrowprops=dict(ec='r',lw = 4))



                    if np.max(cclemon[np.isfinite(cclemon)])/chis[0] >2.5: #if the max is not at least twice the initial then dont break axis        
                        # From https://matplotlib.org/examples/pylab_examples/broken_axis.html
                        d = .015  # how big to make the diagonal lines in axes coordinates
                        # arguments to pass to plot, just so we don't keep repeating them
                        kwargs = dict(transform=ax44.transAxes, color='k', clip_on=False)
                        ax44.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
                        ax44.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

                        kwargs.update(transform=ax4.transAxes)  # switch to the bottom axes
                        ax4.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
                        ax4.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

                        ax44.set_title('Chi-square value vs Datapoints removed', fontsize=titlefont)
                        ax4.set_xlabel('Datapoints removed', fontsize=axisfont)
                        ax4.set_ylabel('Chi-square Value', fontsize=axisfont)
                        ax4.yaxis.set_label_coords(1.06,1)
                        ax44.tick_params(axis='both',labelsize=axisfont)
                    else:
                        ax4.set_title('Chi-square value vs Datapoints removed', fontsize=titlefont)
                        ax4.set_xlabel('Datapoints removed', fontsize=axisfont)
                        ax4.set_ylabel('Chi-square value', fontsize=axisfont)
                else:
                    ax4.text(0.35,0.5,'Nothing to see here', fontsize=titlefont)
            elif graphs == False:
                twostd = meanf+2*stdf

                #get cutoff
                if count == 0:
                    if n_comp > 1:
                        xf = group_div[0]
                        classif = 2
                    else:
                        pass
                        xf = None
                        classif = 1
                elif condition == 1: #remove 1 point means removeing 0 index need to zero index
                    xf = datas[remove][0]
                    classif = 21
                elif condition == 2:
                    xf = np.sort(np.concatenate([group_div,datas[remove]]))
                    classif = 3
                elif condition == 3:
                    xf = datas[-remove][0]
                    classif = 21
                elif condition == 4:
                    xf = np.sort(np.concatenate([group_div,datas[-remove]]))
                    classif = 3
                elif condition == 5:
                    xf = group_div[0]
                    classif = 2

            #get categories
                cutoff=xf
                try:
                    if len(cutoff) == 1:
                        cutoff = cutoff[0]
                except:
                    pass

                if sum(cutoff == None)==1:
                    categories=np.ones(input_data.shape[1]).tolist()
                elif isinstance(cutoff,float) | isinstance(cutoff,np.integer):

                    #determine patients
                    low_exp_group = input_data.columns[(input_data<cutoff).loc[ID]]
                    high_exp_group = input_data.columns[(input_data>=cutoff).loc[ID]]

                    if calc_back == False:

                        #determine datapoints
                        x_lowexp = pd.DataFrame(input_data.loc[:,low_exp_group])
                        x_highexp = pd.DataFrame(input_data.loc[:,high_exp_group])




                        if calc_backpara == True:
                            true_posh = x_highexp.columns[(x_highexp>twostd).loc[ID]]
                            categories = [2 if x in true_posh else 1 for x in input_datanormcat]
                        else:
                            categories = [2 if x in x_highexp.columns else 1 for x in input_datanormcat]

                elif len(cutoff) ==2:
                    #determine patients
                    low_exp_group = input_data.columns[(input_data<cutoff[0]).loc[ID]]
                    med_exp_group = input_data.columns[(cutoff[0] <= input_data.loc[ID]) & (input_data.loc[ID] < cutoff[1])]
                    high_exp_group = input_data.columns[(input_data>=cutoff[1]).loc[ID]]

                    if calc_back == False:

                        #determine datapoints
                        x_lowexp = pd.DataFrame(input_data.loc[:,low_exp_group])
                        x_medexp = pd.DataFrame(input_data.loc[:,med_exp_group])
                        x_highexp = pd.DataFrame(input_data.loc[:,high_exp_group])



                        if calc_backpara == True:
                            true_posh = x_highexp.columns[(x_highexp>twostd).loc[ID]]

                            true_posm = x_medexp.columns[((x_medexp>twostd).loc[ID])]
                            categories = [3 if x in true_posh else 2 if x in true_posm else 1 for x in input_datanormcat]
                        else:
                            categories = [3 if x in x_highexp.columns else 2 if x in x_medexp.columns else 1 for x in input_datanormcat]    

            if calc_back == True:
                return means, np.sqrt(covars), xf

            else:
                if chis == []:
                    return [means, covars, xf], classif, categories, bimodalc
                else:
                    return [means, covars, xf], classif, categories, chis[-1]
            plt.show()

        elif log2transform == False:

                # -------------------------------------------PLOTTTTTTTTTTTTTTTTT------------------------------------------

            x_axis = np.sort(np.array([b for a in datanorm[np.nonzero(datanorm)].reshape(-1,1) for b in a])) #reset x_axis for plotting

            if graphs == True:
                f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=[15,10])

                axisfont = 11
                titlefont = 12.5

                a2 = ax1.twinx()
                #plot 
            #         a2.set_xscale('log',basex=2)
                if n_comp > 1:
                    a2.plot(xs,expected,'--k', color='r',alpha=.5)
                    a2.plot(xs,expectedt,'--k', color='r',alpha=.5)
                    a2.plot(xs,expectedboth, '-k', color='b',alpha=.5)
                else:
                    a2.plot(xs,expected,'-k', color='b',alpha=.5)

                a2.set_ylim(bottom=0)
                a2.set_yticks([])

                #plot histogram of original data
                ax1.set_title('Data GMM Fit of :' + ' '+ ID, fontsize=titlefont)

            #     use this for frequency instead of counts
            #     n, bins, patches = ax1.hist(datanorm,weights=np.ones(len(datanorm))/len(datanorm), 
            #              bins = np.logspace(np.log2(np.min(x_axis)),np.log2(np.max(x_axis)), num=nums,base=2),histtype='barstacked',alpha=0.5, ec='black')

                n, bins, patches = ax1.hist(datanorm, nums,histtype='barstacked',alpha=0.5, ec='black')    
                ax1.set_ylabel('Number of Samples',fontsize=axisfont)
                ax1.set_xlabel('Expression Value',fontsize=axisfont)

                ax1.tick_params(axis='both',labelsize=axisfont)
                a2.tick_params(axis='both',labelsize=axisfont)
                ax1.grid('on')

                if meanf < 10:
                    twostd = 2**(meanf+2*stdf)
                    filt = 2**filt
                else:
                    twostd = meanf+2*stdf
                if calc_back == False and calc_backpara == True:
                    #color potential background, x < threshold
                    if sum(datanorm < 2**twostd) >1 :
                        ind = [i for i, y in enumerate(bins) if y <= 2**(meanf+2*stdf)]
                        if len(ind)>1:
                            for i in range(ind[0],ind[-1]):
                                patches[i].set_facecolor('orange')
                    if np.min(datanorm) < filt:
                        ind = [i for i, y in enumerate(bins) if y <= filt]
                        if len(ind)>1:
                            for i in range(ind[0],ind[-1]):
                                patches[i].set_facecolor('r')

                    background_p = mpatches.Patch(facecolor='lightcoral',ec = 'black',label='Background Threshold')
                    background_p2 = mpatches.Patch(facecolor='gold',ec = 'black',label='Two \u03C3 Background Threshold')

                    a2.legend(handles=[background_p, background_p2])

                if find_target == True:
                    n2, bins2, patches2 = ax1.hist(input_datanorm[cell_lines].loc[ID], nums,hatch='///',facecolor='b',
                                                   histtype='barstacked',alpha=0.5, ec='black') 
                    highlight = input_datanorm[cell_lines].loc[ID]
                    ind = []
                    highlight_p = mpatches.Patch(hatch='///', facecolor = 'w',edgecolor='black',label='Sample(s) of Interest') #add to legend
                    a2.legend(handles=[background_p, background_p2,highlight_p])

                    n2, bins2, patches2 = ax1.hist(input_datanorm[cell_lines].loc[ID],nums, hatch='///',facecolor='b',
                                                   histtype='barstacked',alpha=0.5, ec='black') 
                    #color in with threshold
                    if sum(datanorm < 2**twostd) >1 :
                        ind = [i for i, y in enumerate(bins2) if y <= 2**(meanf+2*stdf)]
                        if len(ind)>1:
                            for i in range(ind[0],ind[-1]):
                                patches2[i].set_facecolor('orange')
                    if np.min(datanorm) < 2**filt:
                        ind = [i for i, y in enumerate(bins2) if y <= 2**filt]
                        if len(ind)>1:
                            for i in range(ind[0],ind[-1]):
                                patches2[i].set_facecolor('r')

                    i = 0 #python zero index
                    ind=[] #zero it
                    for x, y in zip(bins[:-1], np.roll(bins,-1)[:-1]): #find if target value is between bins
                        for z in highlight:
                            if x<=z<=y:
                                ind.append(i)
                        i+=1

                    #show number in that bin
                    i = 0 #python zero index
                    values = pd.value_counts(ind).index
                    counted = list(pd.value_counts(ind))
                    for i, v in enumerate(counted):
                        x = bins[:-1]
                        y = np.roll(bins,-1)[:-1]
                        ax1.text((x[values[i]]+y[values[i]])/2, n[values[i]] + max(n)/50, '(%s)'%v,
                                 horizontalalignment='center', fontsize=axisfont)


                #deem the output unimodal or bimodal whether chi-square method was applied
                intersections = 2**group_div

                if count == 0:
                    if n_comp > 1:
                        for a in intersections: ax1.axvline(a, linestyle='--', c='g')
                        print('Bimodal, Cutoff Threshold:', group_div)
                        xf = group_div[0]
                        classif = 2
                    else:
                        pass
                        print('Unimodal')
                        xf = None
                        classif = 1
                elif condition == 1: #remove 1 point means removeing 0 index need to zero index
                    ax1.axvline(datas[remove][0], linestyle='--', c='g')
                    print('Removed %s datapoints from the left tail, Two Categories by Chi-square Method, Cutoff Threshold: %s' %(remove,datas[remove][0]))
                    xf = datas[remove][0]
                    classif = 21
                elif condition == 2:
                    ax1.axvline(datas[remove][0], linestyle='--', c='g')
                    for a in intersections: ax1.axvline(a, linestyle='--', c='g')
                    print('Removed %s datapoints from the left tail, Three Categories by Chi-square Method, Cutoff Threshold: %s' %(remove,np.sort(np.concatenate([group_div,datas[remove]]))))
                    xf = np.sort(np.concatenate([group_div,datas[remove]]))
                    classif = 3
                elif condition == 3:
                    remd = remove+1 #not zero indexed coming from the other side
                    ax1.axvline(datas[-remd][0], linestyle='--', c='g')
                    print('Removed %s datapoints from the right tail, Two Categories by Chi-square Method, Cutoff Threshold: %s'% (remove,datas[-remove][0]))
                    xf = datas[-remove][0]
                    classif = 21
                elif condition == 4:
                    remd = remove+1 
                    ax1.axvline(datas[-remd][0], linestyle='--', c='g')
                    for a in intersections: ax1.axvline(a, linestyle='--', c='g')
                    print('Removed %s datapoints from the right tail, Three Categories by Chi-square Method, Cutoff Threshold: %s'%(remove,np.sort(np.concatenate([group_div,datas[-remove]]))))
                    xf = np.sort(np.concatenate([group_div,datas[-remove]]))
                    classif = 3
                elif condition == 5:
                    for a in intersections: ax1.axvline(a, linestyle='--', c='g')
                    print('Bimodal cannot be helped by Chi-square Method, Cutoff Threshold:', group_div)
                    xf = group_div[0]
                    classif = 20

            # ----------------------------------------------------------------------------------------------------------------------
                #BIC and AIC graph
                ax2.set_title('BIC Graph of :' + ' '+ ID, fontsize=titlefont)
                ax2.tick_params(axis='both',labelsize=axisfont)

                if condition != 0 and condition !=5:
                    xp=np.arange(0,len(biclemon))
                    length = len(biclemon[0])
                    ax2.set_ylim((0.8,2.2))
                    ax2.set_yticks([1,2])
                    ax2.set_ylabel('n_components',fontsize=axisfont)
                    ax2.set_xlabel('Datapoints Removed',fontsize=axisfont);

                    ax22 = ax2.twinx() #plot BIC value
                    ax22.tick_params(axis='both',labelsize=axisfont)

                    if  length == 1:
                        ax2.plot(xp,biclemon,'--',c ='grey')

                        color = ['b','r']
                        for zz in range(0,len(realbic[0])):
                            bicbag = []
                            for bics in realbic:
                                bicbag.append(bics[zz])
                            ax22.plot(xp,bicbag,'-',c = color[zz])

                        b1 = mlines.Line2D([], [], color=color[0], label='1 n_comp w/ BIC')
                        b2 = mlines.Line2D([], [], color=color[1], label='2 n_comp w/ BIC')
                        b3 = mlines.Line2D([], [], color='grey', label='n_comp w/ Lower BIC Score')
                        ax22.legend(handles=[b1,b2,b3])
                        ax22.set_ylabel('BIC Score',fontsize=axisfont)
                    else:
                        color = ['grey','black']
                        for zz in range(0,length):
                            bicbags = []
                            for bics in biclemon:
                                bicbags.append(bics[zz])
                            ax2.plot(xp,bicbags,'--',c = color[zz])

                        colors = ['deepskyblue','dodgerblue','tomato','lightcoral']

                        newbic = [np.concatenate(x) for x in realbic]
                        for zz in range(0,len(newbic[0])):
                            bicbag = []
                            for bics in newbic:
                                bicbag.append(bics[zz])
                            ax22.plot(xp,bicbag,'-',c = colors[zz])

                        bb1 = mlines.Line2D([], [], color=colors[0], label='Left Tail w/ 1 n_comp BIC')
                        bb2 = mlines.Line2D([], [], color=colors[1], label='Left Tail w/ 2 n_comp BIC')
                        bb3 = mlines.Line2D([], [], color=colors[2], label='Right Tail w/ 1 n_comp BIC')
                        bb4 = mlines.Line2D([], [], color=colors[3], label='Right Tail w/ 2 n_comp BIC')  
                        b1 = mlines.Line2D([], [], color=color[0], label='Left tail n_comp w/ Lower BIC')
                        b2 = mlines.Line2D([], [], color=color[1], label='Right tail n_comp w/ Lower BIC')
                        ax22.legend(handles=[bb1,bb2,bb3,bb4,b1,b2])
                        ax22.set_ylabel('BIC Score',fontsize=axisfont)
                else:
                    ax2.bar(n_components, BIC, width = 0.2, label='BIC',align='center',ec='b')
                    ax2.set_xticks([1,2])
                    ax2.set_ylim(bottom = np.min(BIC)-5, top = np.max(BIC)+2)
                    ax2.set_ylabel('BIC Score',fontsize=axisfont)
                    ax2.set_xlabel('n_components',fontsize=axisfont);

            # ----------------------------------------------------------------------------------------------------------------------
                #plot number of samples in each category

                ax3.grid('on')
                ax3.tick_params(axis='both',labelsize=axisfont)
                cutoff = xf
                try:
                    if len(cutoff) == 1:
                        cutoff = cutoff[0] 
                except:
                    pass

                if sum(cutoff == None)==1:
                    ax3.text(0.35,0.5,'Unimodal', fontsize=titlefont)
                    categories=np.ones(input_datanorm.shape[1]).tolist()
                elif  isinstance(cutoff,float)  | isinstance(cutoff, np.integer):

                    if calc_back == False:
                        #determine patients
                        low_exp_group = input_datanorm.columns[(input_datanorm<cutoff).loc[ID]]
                        high_exp_group = input_datanorm.columns[(input_datanorm>=cutoff).loc[ID]]
                    else:
                        #determine datapoints
                        low_exp_group = datanorm[datanorm < cutoff]
                        high_exp_group = datanorm[datanorm >= cutoff]

                    tol = len(low_exp_group) +len(high_exp_group)
                    print(' No. in Low Expression Group: %s (%s%%)' %(len(low_exp_group),np.round((len(low_exp_group)/tol)*100,2)),'\n', 
                      'No. in High Expression Group: %s (%s%%)' %(len(high_exp_group), np.round((len(high_exp_group)/tol)*100,2)),'\n',
                      'Number of Total:', tol)

                    y = np.round([len(low_exp_group),len(high_exp_group)],2)
                    y_p = np.round([len(low_exp_group)*100/tol,len(high_exp_group)*100/tol],2)
                    x = np.arange(2)

                    ax3.bar(x,y,width=0.2,align='center',ec='b')
                    ax3.set_xticks(x)
                    ax3.set_xticklabels(['Low','High'],fontsize=axisfont)
                    ax3.set_title('Expression Categories of %s'%ID,fontsize=titlefont)
                    ax3.set_ylabel('Number of Samples',fontsize=axisfont)

                    if calc_back == False:
                        #determine datapoints
                        x_lowexp = pd.DataFrame(input_datanorm.loc[:,low_exp_group])
                        x_highexp = pd.DataFrame(input_datanorm.loc[:,high_exp_group])




                        if calc_backpara == True:
                            true_posh = x_highexp.columns[(x_highexp>twostd).loc[ID]]
                            categories = [2 if x in true_posh else 1 for x in input_datanormcat]
                            y2 = np.round([len([]),len(true_posh)],2)

                            ax3.bar(x,y2,width=0.2,align='center',ec='b',facecolor='g')

                            background_d = mpatches.Patch(facecolor='green',ec = 'black',
                                                          label='True Positive(s): %s samples'%(len(true_posh)))
                            ax3.legend(handles=[background_d])
                        else:
                            categories = [2 if x in x_highexp.columns else 1 for x in input_datanormcat]





                    for i, v in enumerate(y_p):
                        ax3.text(x[i]-0.05, y[i] + max(y)/100, str(v)+'%', fontsize=axisfont)

                elif len(cutoff) ==2:
                    if calc_back == False:
                        #determine patients

                        low_exp_group = input_datanorm.columns[(input_datanorm<cutoff[0]).loc[ID]]
                        med_exp_group = input_datanorm.columns[(cutoff[0] <= input_datanorm.loc[ID]) & (input_datanorm.loc[ID] < cutoff[1])]
                        high_exp_group = input_datanorm.columns[(input_datanorm>=cutoff[1]).loc[ID]]

                    else:
                        #determine datapoints
                        low_exp_group = datanorm[datanorm < cutoff[0]]
                        med_exp_group = datanorm[(cutoff[0] <= datanorm) & (datanorm < cutoff[1])]
                        high_exp_group = datanorm[datanorm >= cutoff[1]]

                    tol = len(low_exp_group) +len(high_exp_group)+len(med_exp_group)
                    print(' No. in Low Expression Group: %s (%s%%)' %(len(low_exp_group),np.round((len(low_exp_group)/tol)*100,2)),'\n',
                          'No. in Med Expression Group: %s (%s%%)' %(len(med_exp_group),np.round((len(med_exp_group)/tol)*100,2)),'\n',
                      'No. in High Expression Group: %s (%s%%)' %(len(high_exp_group), np.round((len(high_exp_group)/tol)*100,2)),'\n',
                      'Number of Total:', tol)

                    y = np.round([len(low_exp_group),len(med_exp_group),len(high_exp_group)],2)
                    y_p = np.round([len(low_exp_group)*100/tol,len(med_exp_group)*100/tol,len(high_exp_group)*100/tol],2)
                    x = np.arange(3)


                    ax3.bar(x,y,width=0.2,align='center',ec='b')
                    ax3.set_xticks(x)
                    ax3.set_xticklabels(['Low','Med','High'],fontsize=axisfont)
                    ax3.set_title('Expression Categories of %s'%ID,fontsize=titlefont)
                    ax3.set_ylabel('Number of Samples', fontsize=axisfont)

                    if calc_back == False:

                        #determine datapoints
                        x_lowexp = pd.DataFrame(input_datanorm.loc[:,low_exp_group])
                        x_medexp = pd.DataFrame(input_datanorm.loc[:,med_exp_group])
                        x_highexp = pd.DataFrame(input_datanorm.loc[:,high_exp_group])



                        if calc_backpara == True:
                            true_posh = x_highexp.columns[((x_highexp>twostd).loc[ID])]
                            true_posm = x_medexp.columns[((x_medexp>twostd).loc[ID])]
                            categories = [2 if (x in true_posh) | (x in true_posm) else 1 for x in input_datanormcat]
                            y2 = np.round([len([]),len(true_posm),len(true_posh)],2)

                            ax3.bar(x,y2,width=0.2,align='center',ec='b',facecolor='g')

                            background_d = mpatches.Patch(facecolor='green',ec = 'black',
                                                          label='True Positive(s) = %s samples'%(len(true_posh)+len(true_posm)))
                            ax3.legend(handles=[background_d])

                        else:
                            categories = [2 if (x in x_highexp.columns) | (x in x_medexp.columns) else 1 for x in input_datanormcat]


                    for i, v in enumerate(y_p):
                        ax3.text(x[i]-0.08, y[i]+max(y)/100, str(v)+'%',fontsize=axisfont)

            # ----------------------------------------------------------------------------------------------------------------------
                cclemon = np.array(cclemon)
                #plot chisquare value
                ax4.tick_params(axis='both',labelsize=axisfont)
                from mpl_toolkits.axes_grid1 import make_axes_locatable   

                if condition != 0 and condition !=5:
                    #chisquare value graph
                    xp = np.arange(0,len(cclemon))
                    length = len(cclemon[0])
                    if length == 2:

                        color = ['r','b']

                        l1 = mlines.Line2D([], [], color=color[0], label='Fitting w/ 1 n_comp')
                        l2 = mlines.Line2D([], [], color=color[1], label='Fitting w/ 2 n_comp')
                        dofnum = mpatches.Patch(color='none', label='Degrees of Freedom: %s'%dof)
                        arrows = mpatches.FancyArrowPatch(0.1,0.2,color = 'r' ,label = 'Lower X\N{SUPERSCRIPT TWO} Found' )

                        if np.max(cclemon[np.isfinite(cclemon)])/chis[0] >2.5: #if the max is not at least twice the initial then dont break axis

                            divider = make_axes_locatable(ax4)
                            ax44 = divider.new_vertical(size="100%", pad=0.1,) #break the axis to show lower chi-square
                            f.add_axes(ax44)

                            ax4.set_ylim([0, chis[0]*1.1])
                            ax4.spines['top'].set_visible(False)
                            ax44.set_ylim([np.mean(cclemon[np.isfinite(cclemon)]),np.max(cclemon[np.isfinite(cclemon)])])
                            ax44.set_xticks([])
                            ax44.spines['bottom'].set_visible(False)
                            ax44.legend(handles=[l1,l2,arrows,dofnum])

                        else:
                            ax4.legend(handles=[l1,l2,arrows,dofnum])

                        for zz in range(0,length):
                            lemonbags=[]
                            for lemons in cclemon:
                                lemonbags.append(lemons[zz])
                            ax4.plot(xp,lemonbags,'-',c=color[zz])
                            if np.max(cclemon[np.isfinite(cclemon)])/chis[0] >2.5: #if the max is not at least twice the initial then dont break axis
                                ax44.plot(xp,lemonbags,'-',c=color[zz])

                        highlight = [[x,y] for x,y in zip(rem[2:],chis[1:])]

                        for h in highlight:
                            ax4.annotate('(%s)'%h[0], ha='center',fontsize = axisfont, xy=(h[0], h[1]), xytext=(0, 25), textcoords='offset points', 
                                        arrowprops=dict(ec='r',lw= 4))



                    else:            
                        highlight = [[x,y] for x,y in zip(rem[2:],chis[1:])]
                        color = ['r','b','g','orange']
                        l1 = mlines.Line2D([], [], color=color[0], label='Left Tail w/ 1 n_comp')
                        l2 = mlines.Line2D([], [], color=color[1], label='Left Tail w/ 2 n_comp')
                        r1 = mlines.Line2D([], [], color=color[2], label='Right Tail w/ 1 n_comp')
                        r2 = mlines.Line2D([], [], color=color[3], label='Right Tail w/ 2 n_comp')
                        arrows = mpatches.FancyArrowPatch(0.1,0.2,color = 'r' ,label = 'Lower X\N{SUPERSCRIPT TWO} Found' )
                        dofnum = mpatches.Patch(color='none', label='Degrees of Freedom: %s'%dof)

                        if np.max(cclemon)/chis[0] >2.5: #if the max is not at least twice the initial then dont break axis
                            divider = make_axes_locatable(ax4)
                            ax44 = divider.new_vertical(size="100%", pad=0.1) #break the axis to show lower chi-square
                            f.add_axes(ax44)

                            ax4.set_ylim([0, chis[0]*1.1])
                            ax4.spines['top'].set_visible(False)
                            ax44.set_ylim([np.mean(cclemon[np.isfinite(cclemon)]), np.max(cclemon[np.isfinite(cclemon)])])
                            ax44.tick_params(bottom="off", labelbottom='off')
                            ax44.spines['bottom'].set_visible(False)
                            ax44.legend(handles=[l1,l2,r1,r2,arrows,dofnum])
                        else:
                            ax4.legend(handles=[l1,l2,r1,r2,arrows,dofnum])

                        for zz in range(0,length):
                            lemonbags=[]
                            for lemons in cclemon:
                                lemonbags.append(lemons[zz])
                            ax4.plot(xp,lemonbags,'-',c=color[zz])
                            if np.max(cclemon[np.isfinite(cclemon)])/chis[0] >2.5: #if the max is not at least twice the initial then dont break axis
                                ax44.plot(xp,lemonbags,'-',c=color[zz])

                        for h in highlight:
                            ax4.annotate('(%s)'%h[0], ha='center',fontsize = axisfont, xy=(h[0], h[1]), xytext=(0, 25), textcoords='offset points', 
                                        arrowprops=dict(ec='r',lw = 4))



                    if np.max(cclemon[np.isfinite(cclemon)])/chis[0] >2.5: #if the max is not at least twice the initial then dont break axis        
                        # From https://matplotlib.org/examples/pylab_examples/broken_axis.html
                        d = .015  # how big to make the diagonal lines in axes coordinates
                        # arguments to pass to plot, just so we don't keep repeating them
                        kwargs = dict(transform=ax44.transAxes, color='k', clip_on=False)
                        ax44.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
                        ax44.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

                        kwargs.update(transform=ax4.transAxes)  # switch to the bottom axes
                        ax4.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
                        ax4.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

                        ax44.set_title('Chi-square value vs Datapoints removed', fontsize=titlefont)
                        ax4.set_xlabel('Datapoints removed', fontsize=axisfont)

                        ax4.set_ylabel('Chi-square Value', fontsize=axisfont)
                        ax4.yaxis.set_label_coords(1.06,1)
                        ax44.tick_params(axis='both',labelsize=axisfont)
                    else:
                        ax4.set_title('Chi-square value vs Datapoints removed', fontsize=titlefont)
                        ax4.set_xlabel('Datapoints removed', fontsize=axisfont)
                        ax4.set_ylabel('Chi-square value', fontsize=axisfont)
                else:
                    ax4.text(0.35,0.5,'Nothing to see here', fontsize=titlefont)
            elif graphs == False:
                if meanf < 10:
                    twostd = 2**(meanf+2*stdf)
                    filt = 2**filt
                else:
                    twostd = meanf+2*stdf
                #get cutoff
                if count == 0:
                    if n_comp > 1:
                        xf = group_div[0]
                        classif = 2
                    else:
                        pass
                        xf = None
                        classif = 1
                elif condition == 1: #remove 1 point means removeing 0 index need to zero index
                    xf = datas[remove][0]
                    classif = 21
                elif condition == 2:
                    xf = np.sort(np.concatenate([group_div,datas[remove]]))
                    classif = 3
                elif condition == 3:
                    xf = datas[-remove][0]
                    classif = 21
                elif condition == 4:
                    xf = np.sort(np.concatenate([group_div,datas[-remove]]))
                    classif = 3
                elif condition == 5:
                    xf = group_div[0]
                    classif = 2

            #get categories
                cutoff=xf
                try:
                    if len(cutoff) == 1:
                        cutoff = cutoff[0]
                except:
                    pass

                if sum(cutoff == None)==1:
                    categories=np.ones(input_datanorm.shape[1]).tolist()
                elif isinstance(cutoff,float) | isinstance(cutoff,np.integer):

                    #determine patients
                    low_exp_group = input_datanorm.columns[(input_datanorm<cutoff).loc[ID]]
                    high_exp_group = input_datanorm.columns[(input_datanorm>=cutoff).loc[ID]]

                    if calc_back == False:

                        #determine datapoints
                        x_lowexp = pd.DataFrame(input_datanorm.loc[:,low_exp_group])
                        x_highexp = pd.DataFrame(input_datanorm.loc[:,high_exp_group])



                        if calc_backpara == True:
                            true_posh = x_highexp.columns[(x_highexp>twostd).loc[ID]]
                            categories = [2 if x in true_posh else 1 for x in input_datanormcat]
                        else:
                            categories = [2 if x in x_highexp.columns else 1 for x in input_datanormcat]

                elif len(cutoff) ==2:
                    #determine patients
                    low_exp_group = input_datanorm.columns[(input_datanorm<cutoff[0]).loc[ID]]
                    med_exp_group = input_datanorm.columns[(cutoff[0] <= input_datanorm.loc[ID]) & (input_datanorm.loc[ID] < cutoff[1])]
                    high_exp_group = input_datanorm.columns[(input_datanorm>=cutoff[1]).loc[ID]]

                    if calc_back == False:

                        #determine datapoints
                        x_lowexp = pd.DataFrame(input_datanorm.loc[:,low_exp_group])
                        x_medexp = pd.DataFrame(input_datanorm.loc[:,med_exp_group])
                        x_highexp = pd.DataFrame(input_datanorm.loc[:,high_exp_group])



                        if calc_backpara == True:
                            true_posh = x_highexp.columns[((x_highexp>twostd).loc[ID])]
                            true_posm = x_medexp.columns[((x_medexp>twostd).loc[ID])]
                            categories = [3 if x in true_posh else 2 if x in true_posm else 1 for x in input_datanormcat]
                        else:
                            categories = [3 if x in x_highexp.columns else 2 if x in x_medexp.columns else 1 for x in input_datanormcat]


            if calc_back == True:
                return means, np.sqrt(covars), xf

            else:
                if chis == []:
                    return [means, covars, xf], classif, categories, bimodalc
                else:
                    return [means, covars, xf], classif, categories, chis[-1]
            plt.show()
    
    
#run all hits against one
def find_hits(ip,primary):
    """ This function is used to take in pre-computed subcategorized data and calculate the chi-square contingency table 
    of a single gene or probe with all other genes or probes
    
    :param ip: Input subcategorized data with 1 or 2s
    :param primary: The probe or gene used to calculate chi-square contingency table with all other genes
    
    Returns p-value of all matches, and p-value <= 0.05 for all matches
    
    """
    p_val = []
    odds_ratio = []
    ipa = ip.copy().T
    ipa.replace(3,2,inplace=True) #replace string with integer
    ipa = ipa.loc[:, (ipa != 2).any(axis=0)] #after substituting some are then all 2s

    for x in tqdm(ipa.columns): #DONT DO ANYTHING WITH LOCALIZATION
        
        ipan = ipa[ipa[x] != 0] #take out uncertain = 0

        #determine groups 
        d = {}
        for z in ipan[primary].unique():
            for y in np.sort(ipan[x].unique()):
                d['n%s%s'%(z,y)]= len(ipan[(ipan[primary] == z) & (ipan[x] == y)])

        values, keys = list(d.values()),list(d.keys())

        o, p = stats.fisher_exact([[values[0],values[1]],[values[2],values[3]]])
        p_val.append(p)
        odds_ratio.append(o)

        del o, p #free memory
    
    new = pd.DataFrame({'P-value': p_val},index=ipa.columns).sort_values('P-value',ascending=True)

    filtnew = new[(new<0.05)['P-value']]
    return new, filtnew

#find all hits with pvalue >0.05
def run_hits(ip, index, primary):
    """ This function is used to take in pre-computed subcategorized data, pre chosen index, and calculate the chi-square contingency table 
    of a single gene or probe with all other genes or probes
    
    :param ip: Input subcategorized data with 1 or 2s
    :param index: Pre chosen genes or probes to look at association with the primary
    :param primary: The probe or gene used to calculate chi-square contingency table with all other genes
    
    Returns contingency table, p value, r correlation
    
    """

    index = index.drop(primary)
    
    ipa = ip.copy().T
    
    ct = []
    for x in tqdm(index): #DONT DO ANYTHING WITH LOCALIZATION
        
        ipan = ipa[ipa[x] != 0] #take out uncertain = 0

        #determine groups 
        d = {}
        for z in ipan[primary].unique():
            for y in np.sort(ipan[x].unique()):
                d['n%s%s'%(z,y)]= len(ipan[(ipan[primary] == z) & (ipan[x] == y)])

        values, keys = list(d.values()),list(d.keys())
        
        o, p = stats.fisher_exact([[values[0],values[1]],[values[2],values[3]]])

        pe, pp = stats.pearsonr(ipan[primary],ipan[x])
        ct.append([values[0],values[1],values[2],values[3],p,pe])
        print(pd.crosstab(ipan[primary],ipan[x]))
        print('P-value: %s'%p+'\n')
        print('R-value: %s'%pe+'\n')

        del o,p #free memory
    return ct
#look at one gene
def crosstab_table(ip,index,primary):
    """ This function is used to take in pre-computed subcategorized data and calculate the chi-square contingency table 
    of a single gene or probe with a set of pre chosen genes
    
    :param ip: Input subcategorized data with 1 or 2s
    :param index: Pre chosen genes or probes to look at association with the primary
    :param primary: The probe or gene used to calculate chi-square contingency table with all other genes
    
    """
    
    ipa = ip.copy().T

    
    #determine groups 
    d = {}
    for z in ipa[primary].unique():
        for y in np.sort(ipa[index].unique()):
            d['n%s%s'%(z,y)]= ipa[(ipa[primary] == z) & (ipa[index] == y)].index

    values, keys = list(d.values()),list(d.keys())
    print(pd.crosstab(ipa[primary],ipa[index]))
    print('n11: %s'%list(values[0]) + '\n'+
         'n12: %s'%list(values[1])+ '\n'+
         'n21: %s'%list(values[2])+ '\n'+
        'n22: %s'%list(values[3])+ '\n')

