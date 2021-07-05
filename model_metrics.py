#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras.backend as K


# In[2]:


def dice_coefficient(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[-3,-2,-1])
    dn = K.sum(K.square(y_true) + K.square(y_pred), axis=[-3,-2,-1]) + 1e-8
    return K.mean(2 * intersection / dn, axis=[0,1])


# In[5]:


def dice_coeff(y_true, y_pred, i=0):
    y_true=y_true[:,i]
    y_pred=y_pred[:,i]
    intersection = K.sum(K.abs(y_true * y_pred))
    dn = K.sum(K.square(y_true) + K.square(y_pred)) + 1e-8
    return K.mean(2 * intersection / dn)


# In[4]:


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=[-3,-2,-1])
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=[-3,-2,-1])
    
    return true_positives / (possible_positives + 1e-8)


# In[6]:


def sensi(y_true, y_pred, j=0):
    y_true=y_true[:,j]
    y_pred=y_pred[:,j]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + 1e-8)


# In[7]:


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)), axis=[-3,-2,-1])
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)), axis=[-3,-2,-1])
    
    return true_negatives / (possible_negatives + 1e-8)


# In[9]:


def speci(y_true, y_pred, m=0):
    y_true=y_true[:,m]
    y_pred=y_pred[:,m]    
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + 1e-8)


# In[10]:


def loss_gt(e=1e-8):
    """
    Parameters
    ----------
    `e`: Float, optional
        A small epsilon term to add in the denominator to avoid dividing by
        zero and possible gradient explosion.
        
    Returns
    -------
    loss_gt_(y_true, y_pred): A custom keras loss function
        This function takes as input the predicted and ground labels, uses them
        to calculate the dice loss.
        
    """
    def loss_gt_(y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[-3,-2,-1])
        dn = K.sum(K.square(y_true) + K.square(y_pred), axis=[-3,-2,-1]) + e
        
        return 1 - K.mean(2 * intersection / dn, axis=[0,1])
    
    return loss_gt_


# In[ ]:




