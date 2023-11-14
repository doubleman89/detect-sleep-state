import tensorflow as tf
from tensorflow.keras.layers import Input,Conv1D, MaxPooling1D, Dropout, Conv1DTranspose, Concatenate, Layer, LSTM, Bidirectional,Dense
import numpy as np



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import tensorflow as tf
import time, logging, gc
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import roc_auc_score








class BiLSTM(tf.keras.Model):
    """
    Bi-LSTM model"""
    def __init__(self,n_classes,num_blocks=2, n_filters=32):
        super().__init__()
        self.x1 = Bidirectional(LSTM(768, return_sequences=True))   
        self.x21 = Bidirectional(LSTM(512, return_sequences=True))
        self.x22 = Bidirectional(LSTM(512, return_sequences=True))
        self.l2 = Concatenate(axis=2)
        self.x31 = Bidirectional(LSTM(384, return_sequences=True))
        self.x32 = Bidirectional(LSTM(384, return_sequences=True))
        self.l3 = Concatenate(axis=2)
        self.x41 = Bidirectional(LSTM(256, return_sequences=True))
        self.x42 = Bidirectional(LSTM(128, return_sequences=True))
        self.l4 = Concatenate(axis=2)
        self.l5 = Concatenate(axis=2)
        self.x7 = Dense(128, activation='selu')
        self.x8 = Dropout(0.1)

    def call(self,inputs):
        x = inputs
        x1 = self.x1(x)
        x21 = self.x21(x1)
        x22 = self.x22(inputs)
        l2 = self.l2([x21,x22])
        x31 = self.x31(l2)
        x32 = self.x32(x21)
        l3 = self.l3([x31,x32])
        x41 = self.x41(l3)
        x42 = self.x42(x32)
        l4 = self.l4([x41,x42])
        l5 = self.l5([x1,l2,l3,l4])
        x7 = self.x7(l5)
        x8 = self.x8(x7)
        return x8
    
    def model(self,shape):
        x = Input(shape=shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
def createModel():   
    with strategy.scope():
    
        input_layer = Input(shape=(train.shape[-2:]))
        x1 = Bidirectional(LSTM(768, return_sequences=True))(input_layer)
        
        x21 = Bidirectional(LSTM(512, return_sequences=True))(x1)
        x22 = Bidirectional(LSTM(512, return_sequences=True))(input_layer)
        l2 = Concatenate(axis=2)([x21, x22])
        
        x31 = Bidirectional(LSTM(384, return_sequences=True))(l2)
        x32 = Bidirectional(LSTM(384, return_sequences=True))(x21)
        l3 = Concatenate(axis=2)([x31, x32])
        
        x41 = Bidirectional(LSTM(256, return_sequences=True))(l3)
        x42 = Bidirectional(LSTM(128, return_sequences=True))(x32)
        l4 = Concatenate(axis=2)([x41, x42])
        
        l5 = Concatenate(axis=2)([x1, l2, l3, l4])
        x7 = Dense(128, activation='selu')(l5)
        x8 = Dropout(0.1)(x7)
        output_layer = Dense(units=1, activation="sigmoid")(x8)
        model = Model(inputs=input_layer, outputs=output_layer, name='DNN_Model')
        
        model.compile(optimizer="adam",loss="binary_crossentropy", metrics=[AUC(name = 'auc')])
    return(model)

