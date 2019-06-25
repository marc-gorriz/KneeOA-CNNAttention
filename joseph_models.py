from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Model


def JosephClass(input_shape=(200, 300, 1)):
    inputs = Input(input_shape, name="input")
    x = Conv2D(32, (11, 11), subsample=(2, 2), border_mode='same', activation='relu', name='Conv1_1')(inputs)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='Pool1')(x)
    x = Conv2D(64, (5, 5), border_mode='same', activation='relu', name='Conv2_1')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='Pool2')(x)
    x = Conv2D(96, (3, 3), border_mode='same', activation='relu', name='Conv3_1')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='Pool3')(x)
    x = Conv2D(128, (3, 3), border_mode='same', activation='relu', name='Conv4_1')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='Pool4')(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='fc5')(x)
    x = Dropout(0.2)(x)
    x = Dense(5, activation='softmax', name='fc6')(x)
    return Model(inputs=inputs, outputs=x)


def JosephExt(input_shape=(200, 300, 1)):
    inputs = Input(input_shape, name="input")
    x = Conv2D(32, (11, 11), subsample=(2, 2), border_mode='same', activation='relu', name='Conv1_1')(inp)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='Pool1')(x)
    x = Conv2D(64, (3, 3), border_mode='same', activation='relu', name='Conv2_1')(x)
    x = Conv2D(64, (3, 3), border_mode='same', activation='relu', name='Conv2_2')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='Pool2')(x)
    x = Conv2D(96, (3, 3), border_mode='same', activation='relu', name='Conv3_1')(x)
    x = Conv2D(96, (3, 3), border_mode='same', activation='relu', name='Conv3_2')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='Pool3')(x)
    x = Conv2D(128, (3, 3), border_mode='same', activation='relu', name='Conv4_1')(x)
    x = Conv2D(128, (3, 3), border_mode='same', activation='relu', name='Conv4_2')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='Pool4')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu', name='fc5')(x)
    x = Dropout(0.3)(x)
    return Model(inputs=inputs, outputs=x)
