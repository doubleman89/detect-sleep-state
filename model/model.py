import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate
import numpy as np

class Unet(tf.keras.Model):

    def __init__(self,input_size, n_filters,n_classes):
        inputs = Input(input_size)
        _n_filters = n_filters
        # Add a conv_block with the inputs of the unet_ model and n_filters
        cblock1 = self._conv_block(inputs,_n_filters)
        _n_filters *=2
        # Chain the first element of the output of each block to be the input of the next conv_block. 
        # Double the number of filters at each new step
        cblock2 = self._conv_block(cblock1[0],_n_filters)
        _n_filters *=2
        cblock3 = self._conv_block(cblock2[0],_n_filters)
        _n_filters *=2
        cblock4 = self._conv_block(cblock3[0],_n_filters,dropout_prob=0.3)
        _n_filters *=2
        cblock5 = self._conv_block(cblock4[0],_n_filters,dropout_prob=0.3,max_pooling=False)
        # Expanding Path (decoding)
        _n_filters /=2
        # Add the first upsampling_block.
        ublock6 = self._upsampling_block(cblock5[0],cblock4[1],_n_filters)
        _n_filters /=2
        ublock7 = self._upsampling_block(ublock6[0],cblock3[1],_n_filters)
        _n_filters /=2        
        ublock8 = self._upsampling_block(ublock7[0],cblock2[1],_n_filters)
        _n_filters /=2                
        ublock9 = self._upsampling_block(ublock8[0],cblock1[1],_n_filters)
        _n_filters /=2   
        conv9 = Conv2D(n_filters,
                    3,
                    activation = 'relu',
                    padding = 'same',
                    kernel_initializer = 'he_normal')(ublock9)   
            # Add a Conv2D layer with n_classes filter, kernel size of 1 and a 'same' padding
        conv10 = Conv2D(n_classes,
                    1,
                    activation = 'relu',
                    padding = 'same')(conv9)                    
        
        model = tf.keras.Model(inputs=inputs,outputs = conv10)
    

    def call(self,inputs,tra)
        return model

    def _double_conv(self,inputs = None, n_filters = 32):
        conv = Conv2D(n_filters,
                    3,
                    activation = 'relu',
                    padding = 'same',
                    kernel_initializer = 'he_normal')(inputs)
        conv = Conv2D(n_filters,
                    3,
                    activation = 'relu',
                    padding = 'same',
                    kernel_initializer = 'he_normal')(conv)
        
        return conv

    def _conv_block(self,inputs = None, n_filters = 32, dropout_prob = 0, max_pooling = True):
        """
        Convolutional downsampling block
        
        Arguments:
            inputs -- Input tensor
            n_filters -- Number of filters for the convolutional layers
            dropout_prob -- Dropout probability
            max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
        Returns: 
            next_layer, skip_connection --  Next layer and skip connection outputs
        """ 
           
        conv = self._double_conv(inputs, n_filters)
        
        # if dropout > 0 add dropout layer
        if dropout_prob >0: 
            conv = Dropout(dropout_prob)(conv)

        # if max pooling = True - add maxPooling
        if max_pooling:
            next_layer = MaxPooling2D(pool_size =(2,2))(conv)
        
        else:
            next_layer = conv

        skip_connection = conv

        return next_layer,skip_connection   
    
    def _upsampling_block(self,expansive_input = None, contractve_input = None, n_filters = 32):
        """
        Convolutional downsampling block
        
        Arguments:
            expansive_input -- from the last layer
            contractve_input -- from corresponding downsampling layer
            n_filters -- Number of filters for the convolutional layers
            
        Returns: 
            next_layer -  Next layer 
        """ 


        conv = Conv2DTranspose(n_filters,
                               3)(expansive_input)
        conv = Concatenate(conv,contractve_input)                              
        next_layer = self._double_conv(conv, n_filters)

        

        return next_layer





 
