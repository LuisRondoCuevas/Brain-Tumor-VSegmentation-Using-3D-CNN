#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np  # For data manipulation
import math
import time
from switch import Switch
import matplotlib.pyplot as plt
#import ipyvolume as ipv
from niwidgets import NiftiWidget
from mpl_toolkits.mplot3d import Axes3D 
from skimage.measure import marching_cubes_lewiner
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# In[1]:


# Define constants
SLICE_X = True
SLICE_Y = True
SLICE_Z = False
sliceIndex = 24


# In[2]:


def predictVolume(inImg, toBin=True):
    (xMax, yMax, zMax) = inImg.shape
    
    outImgX = np.zeros((xMax, yMax, zMax))
    outImgY = np.zeros((xMax, yMax, zMax))
    outImgZ = np.zeros((xMax, yMax, zMax))
    
    cnt = 0.0
    if SLICE_X:
        cnt += 1.0
        for i in range(xMax):
            img = inImg[i,:,:] #[np.newaxis,:,:,np.newaxis]
            tmp = img #[:,:,0]
            outImgX[i,:,:] = tmp #scaleImg(tmp, yMax, zMax)
    if SLICE_Y:
        cnt += 1.0
        for i in range(yMax):
            img = inImg[:,i,:] #[np.newaxis,:,:,np.newaxis]
            tmp = img #[:,:,0]
            outImgY[:,i,:] = tmp #scaleImg(tmp, xMax, zMax)
    if SLICE_Z:
        cnt += 1.0
        for i in range(zMax):
            img = inImg[:,:,i] #[np.newaxis,:,:,np.newaxis]
            tmp = img #[:,:,0]
            outImgZ[:,:,i] = tmp #scaleImg(tmp, xMax, yMax)
            
    outImg = (outImgX + outImgY + outImgZ)/cnt
    if(toBin):
        outImg[outImg>0.5] = 1.0
        outImg[outImg<=0.5] = 0.0
    return outImg


# In[5]:


def show_segmented_image(predImg0, predImg1, predImg2, GTImg0, GTImg1, GTImg2,
                         T1c_img, modality='t1c', show=True):
        '''
        Creates an image of original brain with segmentation overlay
        INPUT   (1) str 'test_img': filepath to test image for segmentation, including file extension
                (2) str 'modality': imaging modelity to use as background. defaults to t1c. options: (flair, t1, t1c, t2)
                (3) bool 'show': If true, shows output image. defaults to False.
        OUTPUT  (1) if show is True, shows image of segmentation results
                (2) if show is false, returns segmented image.
        '''
        modes = {'flair', 't1', 't1c', 't2'}

        with Switch(modality) as case:
            if case('t1'):
                m=0
            if case('t2'):
                m=1
            if case('t1c'):
                m=2
            if case('flair'):
                m=3
            if case.default:
                print('error')
            
        #modes=2
        T1c=T1c_img[m]        
        
        ones = np.argwhere(predImg0== 1)
        threes = np.argwhere(predImg1== 1)
        fours = np.argwhere(predImg2== 1)
        
        onesb = np.argwhere(GTImg0== 1)
        threesb = np.argwhere(GTImg1== 1)
        foursb = np.argwhere(GTImg2== 1)

        sliced_image = T1c.copy()
        sliced_imageb = T1c.copy()

        # change colors of segmented classes
        for i in range(len(ones)):
            sliced_imageb[ones[i][0]][ones[i][1]][ones[i][2]]=118 #118 Enhancing Tumor
        #for i in range(len(twos)):
        #    sliced_image[twos[i][0]][twos[i][1]] = green_multiplier
        for i in range(len(threes)):
            sliced_imageb[threes[i][0]][threes[i][1]][threes[i][2]]=200 #200Edema Peritumoral
        for i in range(len(fours)):
            sliced_imageb[fours[i][0]][fours[i][1]][fours[i][2]]=30 #30Tumor Core
        
        #Ground Truth
        for i in range(len(onesb)):
            sliced_image[onesb[i][0]][onesb[i][1]][onesb[i][2]]=118 #118Enhancing Tumor
        for i in range(len(threesb)):
            sliced_image[threesb[i][0]][threesb[i][1]][threesb[i][2]]=200 #200Edema Peritumoral
        for i in range(len(foursb)):
            sliced_image[foursb[i][0]][foursb[i][1]][foursb[i][2]]=30 #30Tumor Core
            
        if show:
            fig = plt.figure(figsize = (10, 7))
            plt.subplot(1, 3, 1)
            plt.imshow(T1c[80,:,:], cmap='gray')
            plt.axis('off')
            plt.title('pat_1- Post-contrast T1')
            plt.subplot(1, 3, 2)
            plt.imshow(sliced_imageb[80,:,:], cmap='CMRmap', vmin=0, vmax=255)
            plt.axis('off')
            plt.title('pat_1-MedU-net')        
            plt.subplot(1, 3, 3)
            plt.imshow(sliced_image[80,:,:], cmap='CMRmap', vmin=0, vmax=255)
            plt.axis('off')
            plt.title('pat_1-Ground Truth')

        else:
            return sliced_imageb


# In[ ]:




