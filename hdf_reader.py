# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:26:23 2019

@author: tvo
"""

'''
***Requirements:
    python > 3
    gdal, numpy, matplotlib, cartopy library
    
**The script has two main fuctions:
    read_hdf(file): read .hdf file and 
                    returns in values as an 2-dimensional array
                    metadata of the dataset
                    number of bands of the dataset

This script is written to read the .hdf file using gdal library

- gdal can be easily installed using: 
    conda install gdal

- library used to plot the dataset:
    matploblib
    cartopy (conda install cartopy or pip install cartopy)
    
    
- HDF files contain variables, and each variable has attributes that describe the variable.

- There are also "global attributes" that describes the overall file

- Here we are testing for reading the HLS dataset: Harmonized Landsat-8 and Sentinel-1 dataset


- The structure of the dataset is a 3-dimensional dataset, which is categorized by several bands and matrix of the values
of corresponding band. 

###To read the metadata of the dataset:
    dataset= gdal.Open(file)
    meta_dataset = dataset.GetMetadata()
 

###Get subdatasets of the dataset
    subset = dataset.GetSubDatasets()


    
'''

import numpy as np
import gdal




def read_hdf(file):
    '''
    Fuction read_hdf to read the hdf file
    Varibale file is the link to the file need to be read
    file = "path/to/file/HLS.L30.T16SCC.2019001.v1.4.hdf"
    '''
    import gdal
    #Open the file
    dataset = gdal.Open(file)
    
    ##Read metadata of the dataset"
    meta_dataset = dataset.GetMetadata()
    
    ##Get subdatasets:
    sub_dataset = dataset.GetSubDatasets()
    
    band_dataset = []
    data_array = []
    ##Read the dataset as an array
    for i in range(len(sub_dataset)):
        
        ##Extracting the title of each band dataset
        band = sub_dataset[i][0]
        
        ##Read values of each band set and return as an 2-dimensioonal array
        data = gdal.Open(band)
        data_arr = data.ReadAsArray()
        
        ##Append the title of each band into a list 
        band_dataset.append(band)
        
        ##Append the values of each band into a list 
        data_array.append(data_arr)
    
        
        
    
    return data_array, band_dataset, meta_dataset

def qa_code(data, metadata):
    """
    Function to decode the QA layer and create separate cloud layer and add new bands to current dataset
        return values: 0 or 1 (0: no, 1:yes)
        circus: band 8
        cloud: band 9
        adjacent cloud: band 10
        cloud shadow: band 11
        
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    qa_layer = data[-1] ##Get the QA layer product from the dataset
    qa_decode = np.vectorize(np.binary_repr)(qa_layer,width=8) ##Decode the qa layer which is stored as unsigned 8-bit integer
    
    ##Create empty layers for each of the cloud layer
    circus = np.empty_like(qa_layer, dtype=np.int16)
    cloud = np.empty_like(qa_layer, dtype=np.int16)
    adj_could = np.empty_like(qa_layer, dtype=np.int16)
    sha_cloud = np.empty_like(qa_layer, dtype=np.int16)
    
    
    
    ##Loop over the empty cloud layers and add the corresponding bit values to each layers
    for row in range(len(qa_decode)):
        for col in range(len(qa_decode)):
            circus[row][col] = int(qa_decode[row][col][-1])
            cloud[row][col] = int(qa_decode[row][col][-2])
            adj_could[row][col] = int(qa_decode[row][col][-3])
            sha_cloud[row][col] = int(qa_decode[row][col][-4])
    
    ##Stack existing cloud layers to a 3-D arrays with order of : circus, cloud, adj_cloud, sha_cloud
    cloud_set = np.dstack((circus, cloud, adj_could, sha_cloud))  
    band = list("circus","cloud","adjection cloud","cloud shadow")

      
    fig, axes = plt.subplots()
    axes = (ax1, ax2, ax3, ax4)
    for ax in axes:
        for i in range(len(cloud_set.shape[-1])):
            ax.imshow(cloud_set[:,:,i],cmap="gray")
            ax.set_title(band[i])
    
    plt.show()
            
        
        
    
    #if metadata[""]
            
    
    
    
    
    return data_cloud
    
    

def plot_hdf(data,  metadata, bands=None, RGB=None):
    '''
    Documentation
    function plot_hdf to plot the array values have readed from .hdf file as an map 
    Variables:
        data: the dataset 
        bands: the index of bands need to be use. e.g. Band01: 0, Band02: 1,..... 
        
        
    Optional parameters:
        RGB: color composite. The input values must be as an array [band_red, band_green, band_blue]
        RGB=[0,2,5] ==> Red: Band 01; Green: Band 03; Blue: Band 06
    '''
    import matplotlib.pyplot as plt
    
    
    if bands is not None:
        
        fig, ax = plt.subplots()  
        ax.imshow(data[bands])
        title = [metadata["ACCODE"]," Band: ",bands+1]
        print(title)
        ax.set_title(title,size=30)
    
    
    
    if RGB is not None:

        rgb = {'r': data[RGB[0]], 'g':data[RGB[1]], 'b': data[RGB[2]]}
        stack = lambda arr: np.dstack(list(arr.values()))
        data = stack(rgb)
        fig, ax = plt.subplots()
        ax.imshow(data)
        title = [metadata["ACCODE"]," RGB: ",RGB]
        ax.set_title(title,size=30)


    
    return None


if __name__ == '__main__':
    
    import click

    with click.progressbar(range(1000000)) as bar:
        for i in bar:
            pass
        
        
    file = 'C:/Users/tvo/Downloads/HLS.L30.T16SCC.2019001.v1.4.hdf'
    
    ##Read .hdf file
    data, bands, metadata = read_hdf(file)
    
    ##Set the boundary of the map based on the values in metadata
    bound_lons = [float(metadata["ULX"]) , 
                  float(metadata["ULX"]) + float(metadata["NCOLS"]) * float(metadata["SPATIAL_RESOLUTION"])]
    
    bound_lats = [float(metadata['ULY']) , 
                  float(metadata["ULY"]) - float(metadata["NROWS"]) * float(metadata["SPATIAL_RESOLUTION"])]
    
    ##Select the band that user would like to create a
    rgb = {'r': data[1], 'g':data[4], 'b': data[3]}
    
    
    
    ##Plot the data Band01
    plot_hdf(data, metadata, bands=0)
    
    ##Plot the data RGB = 1,2,3
    plot_hdf(data, metadata, RGB=[2,6,8])
    
    
    ##Create cloud band 
    data_cloud = qa_code(data,metadata)
 
    
    pass

