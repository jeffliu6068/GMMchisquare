# GMMchisquare

This is a package for Gaussian Mixture Modeling (GMM) using the chi-square protocol to subcategorize gene expression data. The method is based on understanding mixtures of Guassian distribution and non-normally distributed tails. This makes creates a wide application for this protocol for any distribution-based pattern recognition for subcategorizing data. 

Clear a-priori knowledge of expected distinct sub populations due to underlying mechanism is ideal for this method. The original use case is for gene expression data where the observed pattern of two sub populations due to mutated and normal population and their respective expression pattern clustering in distinct distributions gave rise to this protocol. Let's take a look at how we better use this package for gene expression analysis. 

This is a python package. I will be detailing a step-by-step for those who have never used python:

## Getting Started 

Download Anaconda at https://www.anaconda.com/distribution/

After downloading, open terminal (Mac) or cmd (Windows) by searching for them respectively depending on your operating system. Open Jupyter Notebook either by entering 'jupyter notebook' in the terminal/cmd or opening it using the anaconda application downloaded. This will open a visual shell script using your default browser. 

Create a new script that can be edited by going to your desired folder --> new --> python 3 notebook located on the top right corner

### Download Package

Download this package by:
```
!pip install git+https://github.com/jeffliu6068/GMMchisquare.git
```
or 
```
!pip install GMMchisquare
```

### Import

Once installed, import the package by: 

```
import GMMchisquare as GMM
```

## Available Tools in the GMMchisquare Package

There are several functions that is included in the package: 


### GMM.probe_filter 

GMM.probe_filter is used to filter probes or genes based on the background threshold. For example:

```
input_dataf = probe_filter(input_data_cancer,log2transform=True,filt=-0.829)
```

#### Input

input_data_cancer: input data with genes (row) x samples (columns)

log2transform: perform log2-transformation on the data

filt: level of background expression log2 transformed (2^-0.829) to match with the parameter log2-transform = True 

#### Returns Output 

input_dataf: returns the filtered data


### GMM.GMM_modelingt

GMM.GMM_modelingt is the main function which uses GMM and chi-square fit protocol to decipher the underlying sub populations. For example:

### *Use Case1: calculating background threshold*

```
means, std, filt = GMM_modelingt('TCGA Colorectal Cancer' ,input_data_cancer,log2transform=True,
                      ,calc_back=True, calc_backpara= False)
```
#### Input

'TCGA Colorectal Cancer': ID in this case will be automatically used as the title for your output graphs

input_data_cancer: input data with genes (row) x samples (columns)
             
log2transform: perform log2-transformation on the data

calc_back: True to calculate background threshold

calc_backpara: False to turn off since we are calculating for background threshold to draw difference between noise expression level and negative

#### Returns Output

means: Mean of identified distributions

std: Standard deviation of identified distributions

filt: Cutoff between the distributions


### *Use case2: Subcategorizing the distribution a single gene*

```
gene = 'TGFB1'

info, classif, categories,chi = GMM_modelingt(gene,input_dataf,log2transform=True,calc_backpara=True
                                    ,filt=-0.83, meanf= -3.3, stdf = 1.95)
```
#### Input

gene: gene name you're interested looking at

input_dataf: input data with genes (row) x samples (columns), this can be the output of the **GMM.probe_filter** 
                    
log2transform: perform log2-transformation on the data

calc_backpara: use background threshold to draw difference between noise expression level and negative

filt: level of background expression log2 transformed 

meanf: mean of the background distribution

stdf: standard deviation of background distribution 

#### Returns Output

info: mean, covariance, and threshold of identified distribution

classif: classification annotation 

        Classification annotation:
        1: Unimodal distribution
        2: Bimodal distribution
        21: Unimodal distribution + tail
        3: Bimodal distribution + tail
        20: Biomdal distribution with chi-square fit protocol failed to fit 
        
categories: catogries of each sample after GMMchisquare

chi: lowest chi-square goodness of fit value during fitting 

