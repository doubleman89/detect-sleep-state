import tensorflow as tf
from tensorflow.keras.layers import Input,Conv1D, MaxPooling1D, Dropout, Conv1DTranspose, Concatenate, Layer
import numpy as np

class DoubleConv1D(Layer):
    def __init__(self, n_filters = 32,kernel_size = 3 ,strides = 1):
        super().__init__()
        self.conv1 = Conv1D(n_filters,
                    kernel_size = kernel_size,
                    strides = strides,
                    activation = 'relu',
                    padding = 'same',
                    kernel_initializer = 'he_normal')
        self.conv2 = Conv1D(n_filters,
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
        max_pooling -- Use MaxPooling1D to reduce the spatial dimensions of the output volume
    Returns: 
        next_layer, skip_connection --  Next layer and skip connection outputs
    """ 
    def __init__(self, n_filters = 32, dropout_prob = 0.0, max_pooling = True):
        super().__init__()
        self.dconv1 = DoubleConv1D(n_filters)
        self.dropout_prob = dropout_prob
        self.dropout = Dropout(dropout_prob)

        self.max_pooling = max_pooling
        self.max_pooling2d = MaxPooling1D(pool_size =2)

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
        self.conv1D = Conv1DTranspose(n_filters,
                            3, # kernel size
                            strides = 2,
                            padding = 'same')
        self.concat = Concatenate(axis = 2)
        self.dconv1 = DoubleConv1D(n_filters,kernel_size = 3)


    def call(self,expansive_input = None, contractve_input = None):
        x = self.conv1D(expansive_input)
        x = self.concat([x,contractve_input])
        next_layer = self.dconv1(x)
        return next_layer
    
class Unet(tf.keras.Model):
    """
    Unet model"""
    def __init__(self,n_classes,num_blocks=5, n_filters=32):
        super().__init__()
        self.downsampling_blocks  = []
        self._n_filters = n_filters
        for i in range(num_blocks):
            if i ==3:
                self.downsampling_blocks.append(DownsamplingBlock(self._n_filters,dropout_prob=0.3) )    
                self._n_filters *= 2
            #if i ==1:
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
        self.conv1 = Conv1D(self._n_filters,
            3,
            activation = 'relu',
            padding = 'same',
            kernel_initializer = 'he_normal')
        # Add a Conv1D layer with n_classes filter, kernel size of 1 and a 'same' padding
        self.conv2 = Conv1D(n_classes,
            1,
            activation = 'relu',
            padding = 'same')    
    def call(self,inputs):
        x = inputs
        contractive_inputs = []
        for downblock in self.downsampling_blocks:

            d = downblock(x)
            x= d[0]
            contractive_inputs.append(d[1])
        
        # remove the last contractive input from the bottom - not used 
        contractive_inputs.pop()

        for upblock in self.upsampling_blocks:
            contractive_input = contractive_inputs.pop()
            x = upblock(x,contractive_input)
        x = self.conv1(x)

        return self.conv2(x)
    
    def model(self,shape):
        x = Input(shape=shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))





class Unet_Model:

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
        conv9 = Conv1D(n_filters,
                    3,
                    activation = 'relu',
                    padding = 'same',
                    kernel_initializer = 'he_normal')(ublock9)   
            # Add a Conv1D layer with n_classes filter, kernel size of 1 and a 'same' padding
        conv10 = Conv1D(n_classes,
                    1,
                    activation = 'relu',
                    padding = 'same')(conv9)                    
        
        model = tf.keras.Model(inputs=inputs,outputs = conv10)
    

    def _double_conv(self,inputs = None, n_filters = 32):
        conv = Conv1D(n_filters,
                    3,
                    activation = 'relu',
                    padding = 'same',
                    kernel_initializer = 'he_normal')(inputs)
        conv = Conv1D(n_filters,
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
            max_pooling -- Use MaxPooling1D to reduce the spatial dimensions of the output volume
        Returns: 
            next_layer, skip_connection --  Next layer and skip connection outputs
        """ 
           
        conv = self._double_conv(inputs, n_filters)
        
        # if dropout > 0 add dropout layer
        if dropout_prob >0: 
            conv = Dropout(dropout_prob)(conv)

        # if max pooling = True - add maxPooling
        if max_pooling:
            next_layer = MaxPooling1D(pool_size =(2,2))(conv)
        
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


        conv = Conv1DTranspose(n_filters,
                               3)(expansive_input)
        conv = Concatenate(conv,contractve_input)                              
        next_layer = self._double_conv(conv, n_filters)

        

        return next_layer





 
