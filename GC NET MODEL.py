from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D
from keras import backend as bk
from keras import optimizers
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers import Activation

from keras.utils.generic_utils import get_custom_objects
def piecewise5(X):
                   return bk.switch(X < -0.6, (0.01 * X ),
                                   bk.switch(X < -0.2, (0.2 * X ),
                                            bk.switch(X < 0.2, (1 * X ),
                                                     bk.switch(X < 0.6, (1.5 * X ),
                                                              bk.switch(X < 5, (3 * X ), (3 * X )))))) 
    
get_custom_objects().update({'piecewise5': Activation(piecewise5)})




input_shape=X_train.shape[1:]

def custom_network(input_shape):

#   model = Sequential()
#   model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
#   model.add(Conv2D(64, (3, 3), activation='relu'))
#   model.add(MaxPooling2D(pool_size=(2, 2)))
#   model.add(Dropout(0.25))
#   model.add(Flatten())
#   model.add(Dense(128, activation='relu'))
#   model.add(Dropout(0.5))
#   model.add(Dense(num_classes, activation='softmax'))
  
  
    input_img = Input(shape = (30, 30, 3))
    
    conv_1 = Conv2D(64, (3,3), padding='same', activation='piecewise5')(input_img)
    block1_output = GlobalAveragePooling2D()(conv_1)
    max_pool_1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same')(conv_1)
    dropout_1 = Dropout(0.25)(max_pool_1)
    
    
    
    conv_2 = Conv2D(64, (3,3), padding='same', activation='piecewise5')(dropout_1)
    block2_output = GlobalAveragePooling2D()(conv_2)
    max_pool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(conv_2)
    dropout_2 = Dropout(0.01)(max_pool_2)
    
    
    conv_3 = Conv2D(64, (3,3), padding='same', activation='piecewise5')(dropout_2)
    block3_output = GlobalAveragePooling2D()(conv_3)
    
    

   
    
    output = keras.layers.concatenate([block1_output, block2_output, block3_output], axis = 1)
#     output = Flatten()(output)
    output = Dense(64,activation='piecewise5')(output)
    out    = Dense(43, activation='softmax')(output)
    
   

    
    
    
    
    model = Model(inputs = input_img, outputs = out)
    print(model.summary())
    return model
#     input_img = Input(shape = (30, 30, 3))
#     tower_1 = Conv2D(16, (1,1), padding='same', activation='elu')(input_img)
#     tower_x = Conv2D(32, (3,3), padding='same', activation='elu')(tower_1)
#     tower_y = MaxPooling2D(pool_size=(2, 2), padding='same')(tower_x)
#     tower_y = Dropout(0.1)(tower_y)
#     tower_z = Conv2D(32, (1,1), padding='same', activation='elu')(tower_y)
#     tower_a = Conv2D(32, (3,3), padding='same', activation='elu')(tower_z)
#     tower_a = MaxPooling2D(pool_size=(2, 2), padding='same')(tower_a)
#     tower_a = Dropout(0.1)(tower_a)
    
  
#     tower_2 = MaxPooling2D(pool_size=(4, 4), padding='same')(tower_x)
#     tower_2 = Dropout(0.1)(tower_2)
    
#     tower_3 = MaxPooling2D(pool_size=(2, 2), padding='same')(tower_z)
#     tower_3 = Dropout(0.1)(tower_3)
    
    

    
    
    
    
    
    
#     output = keras.layers.concatenate([tower_a, tower_2, tower_3], axis = 1)
#     output = Flatten()(output)

#     out    = Dense(43, activation='softmax')(output)
    
   

    
    
    
    
#     model = Model(inputs = input_img, outputs = out)
#     print(model.summary())
#     return model





sgd=optimizers.SGD(lr=0.001, momentum=0.9, nesterov=False)
model = custom_network(input_shape)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

# #Compilation of the model
# model.compile(
#     loss='categorical_crossentropy', 
#     optimizer='adam', 
#     metrics=['accuracy']
# )
#using ten epochs for the training and saving the accuracy for each epoch
epochs = 20
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs,
validation_data=(X_val, y_val))

# score = model.evaluate(x_test, y_test, verbose=0)
# score2=model.evaluate(x_train,y_train, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# print('Training loss:', score2[0])
# print('Training accuracy:', score2[1])# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

