from keras import backend as K
form keras.layers.core import Lambda

def my_tf_round(x, decimals=7):
    multiplier = K.constant(10**decimals, dtype=x.dtype)
    return K.round(x * multiplier) / multiplier
    
"""
x = Lambda(my_tf_round)(x)
"""
