#!/usr/bin/env python
# coding: utf-8

# In[6]:


from keras.layers import *
from group_norm import GroupNormalization


# In[7]:


def conc_block(back_tensor, input_tensor, conc_filters, conc_kernel):
    
    # Concatenation block
    c1s = Conv3D(filters=conc_filters, kernel_size=(conc_kernel, conc_kernel, conc_kernel), kernel_initializer="he_normal",
               padding="same", data_format='channels_first')(input_tensor)
    #c1s = BatchNormalization()(c1s)
    c1s = Activation("relu")(c1s)
    
    c4s = concatenate([back_tensor, c1s], axis=-4)
    return c4s


# In[8]:


def conv3d_block(input_tensor, n_filters, kernel_size, data_format, groupnorm=True):
    """Construye un bloque convolucional compuesto por dos capas convolucionales, con activación ReLU
    y BatchNormalization si np se especifica lo contrario"""
    name=None
    
    # first layer
    x = Conv3D(filters=n_filters, kernel_size=(kernel_size, kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same", data_format='channels_first')(input_tensor)
    if groupnorm:
        x = GroupNormalization(groups=8,
        axis=1 if data_format == 'channels_first' else 0,
        name=f'GroupNorm_1_{name}' if name else None)(x)
    x = Activation("relu")(x)
    
    # second layer
    x = Conv3D(filters=n_filters, kernel_size=(kernel_size, kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same", data_format='channels_first')(x)
    if groupnorm:
        x = GroupNormalization(groups=8,
        axis=1 if data_format == 'channels_first' else 0,
        name=f'GroupNorm_2_{name}' if name else None)(x)
    x = Activation("relu")(x)
    return x


# In[ ]:




