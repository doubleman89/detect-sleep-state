import tensorflow as tf
from tensorflow.keras.layers import Input,Dropout, Dense


class FCModel(tf.keras.Model):
    def __init__(self,n_classes):
        super().__init__()
        self.x1 = Dense(32,'relu')
        self.x2 = Dense(64,'relu')
        self.x2d = Dropout(0.3)
        self.x3 = Dense(32,'relu')
        self.x4 = Dense(16,'relu')
        self.x4d = Dropout(0.3)
        #self.x5 = Dense(n_classes,'softmax')
        self.x5 = Dense(n_classes,'sigmoid')
        
    def call (self,inputs):
        x = inputs
        x = self.x1(x)
        x = self.x2(x)
        x = self.x2d(x)
        x = self.x3(x)
        x = self.x4(x)
        x = self.x4d(x)
        
        return self.x5(x)
    def model(self,shape):
        x = Input(shape=shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
    