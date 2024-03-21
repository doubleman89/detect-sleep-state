import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, Layer
import numpy as np

class DoubleConv2D(Layer):
    def __init__(self, n_filters = 32,kernel_size = 3 ,strides = (1,1)):
        super().__init__()
        self.conv1 = Conv2D(n_filters,
                    kernel_size = kernel_size,
                    strides = strides,
                    activation = 'relu',
                    padding = 'same',
                    kernel_initializer = 'he_normal')
        self.conv2 = Conv2D(n_filters,
                    kernel_size = kernel_size,
                    strides = strides,
                    activation = 'relu',
                    padding = 'same',
                    kernel_initializer = 'he_normal')
        
    def call(self,inputs):
        x = self.conv1(inputs)
        return self.conv2(x)
        

        
class DownsamplingBlock(Layer):
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
    def __init__(self, n_filters = 32, dropout_prob = 0.0, max_pooling = True):
        super().__init__()
        self.dconv1 = DoubleConv2D(n_filters)
        self.dropout_prob = dropout_prob
        self.dropout = Dropout(dropout_prob)

        self.max_pooling = max_pooling
        self.max_pooling2d = MaxPooling2D(pool_size =(2,2))

    def call(self,inputs):
        x= self.dconv1(inputs)
        # if dropout > 0 add dropout layer
        if self.dropout_prob > 0.0:
            x= self.dropout(x)
        # if max pooling = True - add maxPooling
        if self.max_pooling:
            next_layer = self.max_pooling2d(x)
        else:
            next_layer = x
        skip_connection = x
       # print(next_layer.shape,skip_connection.shape)
        return next_layer,skip_connection


class UpsamplingBlock(Layer):
    def __init__(self, n_filters = 32):
        """
        Convolutional downsampling block
        
        Arguments:
            expansive_input -- from the last layer
            contractve_input -- from corresponding downsampling layer
            n_filters -- Number of filters for the convolutional layers
            
        Returns: 
            next_layer -  Next layer 
        """ 
        super().__init__()
        self.conv2D = Conv2DTranspose(n_filters,
                            (3,3), # kernel size
                            strides = (2,2),
                            padding = 'same')
        self.concat = Concatenate(axis = 3)
        self.dconv1 = DoubleConv2D(n_filters,kernel_size = (3,3))


    def call(self,expansive_input = None, contractve_input = None):
        print(expansive_input.shape)
        x = self.conv2D(expansive_input)
        print(x.shape,contractve_input.shape)
        x = self.concat([x,contractve_input])
        next_layer = self.dconv1(x)
        return next_layer
    
class Unet_Multi(tf.keras.Model):
    """
    Unet model"""
    def __init__(self,n_classes,multi_class_range, num_blocks=5, n_filters=32):
        super().__init__()
        self.downsampling_blocks  = []
        self._n_filters = n_filters
        for i in range(num_blocks):
            if i ==3:
                self.downsampling_blocks.append(DownsamplingBlock(self._n_filters,dropout_prob=0.3) )    
                self._n_filters *= 2
            elif i==4:
                self.downsampling_blocks.append(DownsamplingBlock(self._n_filters,dropout_prob=0.3,max_pooling=False) )    
                self._n_filters /=2
            else:
                self.downsampling_blocks.append(DownsamplingBlock(self._n_filters) )
                self._n_filters *= 2
        self.upsampling_blocks  = []
        for i in range(num_blocks-1):
            self.upsampling_blocks.append(UpsamplingBlock(self._n_filters) )
            self._n_filters /=2
        self.conv1 = Conv2D(self._n_filters,
            3,
            activation = 'relu',
            padding = 'same',
            kernel_initializer = 'he_normal')
        # Add a Conv2D layer with n_classes filter, kernel size of 1 and a 'same' padding
        # softmax classifier output for awake ranges 
        self.conv2_1 = Conv2D(16,
            1,
            activation = 'relu',
            padding = 'same')    
        self.output_multi_awake = Conv2D(multi_class_range,
            1,
            activation = 'softmax',
            padding = 'same') 
        # softmax classifier output for sleep ranges 
        self.conv2_2 = Conv2D(16,
            1,
            activation = 'relu',
            padding = 'same')    
        self.output_multi_sleep = Conv2D(multi_class_range,
            1,
            activation = 'softmax',
            padding = 'same') 
        # binary classifier output for sleep awake 
        self.conv2_3 = Conv2D(16,
            1,
            activation = 'relu',
            padding = 'same')    
        self.output_sleep_awake = Conv2D(n_classes,
            1,
            activation = 'softmax',
            padding = 'same') 
        
    def call(self,inputs):
        x = inputs
        contractive_inputs = []
        for downblock in self.downsampling_blocks:

            d = downblock(x)
            x= d[0]
            contractive_inputs.append(d[1])
        
        # remove the last contractive input from the bottom - not used 
        print(len(contractive_inputs))
        contractive_inputs.pop()
        print(len(contractive_inputs))

        for upblock in self.upsampling_blocks:
            contractive_input = contractive_inputs.pop()
            x = upblock(x,contractive_input)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_3(x)
        x3 = self.conv2_3(x)
        y1 = self.output_multi_awake(x1)
        y2 = self.output_multi_sleep(x2)        
        y3 = self.output_sleep_awake(x3)

        return (y1,y2,y3)
    
    def model(self,shape):
        x = Input(shape=shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))



 
