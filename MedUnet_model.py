#!/usr/bin/env python
# coding: utf-8

# In[13]:


from keras.models import *
from keras.layers import *
from model_blocks import conv3d_block, conc_block


# In[15]:


def build_MedUnet(input_shape, output_channels, n_filters, conc_filters, kernel_size, conc_kernel, dropout, groupnorm):
    """Construye la arquitectura de la red, a través de bloques convolucionales,
    seguidos por MaxPool y Dropout para el camino de contracción. 
    El camino de expansión se compone por una convolución inversa seguida por Dropout y 
    un bloque deconvolucional."""
    # Input Layer
    inp = Input(input_shape);
    
    # contracting path
    c1 = conv3d_block(inp, n_filters=n_filters*1, kernel_size=3, data_format = 'channels_first', groupnorm=groupnorm)
    p1 = MaxPooling3D((2, 2, 2), data_format='channels_first') (c1)
    p1 = SpatialDropout3D(dropout*0.5, data_format='channels_first')(p1)

    c2 = conv3d_block(p1, n_filters=n_filters*2, kernel_size=3,data_format = 'channels_first', groupnorm=groupnorm)
    p2 = MaxPooling3D((2, 2, 2), data_format='channels_first') (c2)
    p2 = SpatialDropout3D(dropout, data_format='channels_first')(p2)

    c3 = conv3d_block(p2, n_filters=n_filters*4, kernel_size=3,data_format = 'channels_first', groupnorm=groupnorm)
    p3 = MaxPooling3D((2, 2, 2), data_format='channels_first') (c3)
    p3 = SpatialDropout3D(dropout, data_format='channels_first')(p3)

    c4 = conv3d_block(p3, n_filters=n_filters*8, kernel_size=3,data_format = 'channels_first', groupnorm=groupnorm)
    #p4 = MaxPooling3D((2, 2, 2), data_format='channels_first') (c4)
    p4 = SpatialDropout3D(dropout, data_format='channels_first')(c4)
    
    c5 = conv3d_block(p4, n_filters=n_filters*16, kernel_size=3,data_format = 'channels_first', groupnorm=groupnorm)
    
    # expansive path
    #u6 = Conv3DTranspose(n_filters*8, (3, 3, 3), strides=(2, 2, 2), padding='same', data_format='channels_first') (c5)
    #u6 = conc_block(u6, c4, n_filters, conc_kernel, batchnorm=True)
    u6 = SpatialDropout3D(dropout, data_format='channels_first')(c5)
    c6 = conv3d_block(u6, n_filters=n_filters*8, kernel_size=3, data_format = 'channels_first', groupnorm=groupnorm)

    u7 = Conv3DTranspose(n_filters*4, (3, 3, 3), strides=(2, 2, 2), padding='same', data_format='channels_first') (c6)
    u7 = conc_block(u7, c3, conc_filters, conc_kernel)
    u7 = SpatialDropout3D(dropout, data_format='channels_first')(u7)
    c7 = conv3d_block(u7, n_filters=n_filters*4, kernel_size=3, data_format = 'channels_first', groupnorm=groupnorm)

    u8 = Conv3DTranspose(n_filters*2, (3, 3, 3), strides=(2, 2, 2), padding='same', data_format='channels_first') (c7)
    u8 = conc_block(u8, c2, conc_filters, conc_kernel)
    u8 = SpatialDropout3D(dropout, data_format='channels_first')(u8)
    c8 = conv3d_block(u8, n_filters=n_filters*2, kernel_size=3, data_format = 'channels_first', groupnorm=groupnorm)

    u9 = Conv3DTranspose(n_filters*1, (3, 3, 3), strides=(2, 2, 2), padding='same', data_format='channels_first') (c8)
    u9 = conc_block(u9, c1, conc_filters, conc_kernel)
    u9 = SpatialDropout3D(dropout, data_format='channels_first')(u9)
    c9 = conv3d_block(u9, n_filters=n_filters*1, kernel_size=3, data_format = 'channels_first', groupnorm=groupnorm)
    
    outputs = Conv3D(output_channels, (1, 1, 1), strides=(1, 1, 1), data_format='channels_first', activation='sigmoid') (c9)
    
    model = Model(inp, outputs=[outputs])
    
    return model


# In[ ]:




