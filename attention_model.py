import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.layers import BatchNormalization, Dense, Lambda
from keras.layers import Conv2D, LocallyConnected2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import concatenate, multiply
from keras.models import Model

from test_utils import get_layer
from .joseph_models import JosephClass, JosephExt


class _BaseModel:
    def __init__(self, name='base_model',
                 core_model=VGG16,
                 att_layers=None,
                 nb_classes=5):
        self.name = name
        self.core_model = core_model
        self.att_layers = att_layers
        self.nb_classes = nb_classes

    @staticmethod
    def up_depth(depth):
        layer = Conv2D(depth, kernel_size=(1, 1), padding='same', activation='linear',
                       use_bias=False, weights=[np.ones(shape=(1, 1, 1, depth))])
        layer.trainable = False
        return layer

    def attention_branch(self, input_layer, name):
        layer = input_layer.get_output_at(0)
        depth = layer.shape[-1].value
        bn = x = BatchNormalization()(layer)
        x = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu')(x)
        x = LocallyConnected2D(1, kernel_size=(1, 1), padding='valid', activation='sigmoid', name=name)(x)
        att = self.up_depth(depth)(x)
        features = multiply([att, bn])
        features = GlobalAveragePooling2D()(features)
        mask = GlobalAveragePooling2D()(att)
        x = Lambda(lambda i: i[0] / i[1])([features, mask])
        return x

    def combine_single(self, att_layers, early_fusion=0):
        x = concatenate(att_layers) if len(att_layers) > 1 else att_layers[0]
        if early_fusion > 0: x = Dense(early_fusion, activation='relu')(x)
        return Dense(self.nb_classes, activation='softmax')(x)

    def combine_multi(self, att_layers):
        return [Dense(self.nb_classes, activation='softmax')(x) for x in att_layers]

    def get_model(self, multi_loss=False, early_fusion=0):
        if len(self.att_layers) > 1:
            assert multi_loss or early_fusion > 0
        else:
            assert not multi_loss or early_fusion == 0
        att_layers = []
        for idx, name in enumerate(self.att_layers):
            input_layer = get_layer(self.core_model, name=name)
            att_layers.append(self.attention_branch(input_layer, name='Att_%d' % idx))
        outputs = self.combine_multi(att_layers) if multi_loss else \
            self.combine_single(att_layers, early_fusion)
        return Model(self.core_model.inputs, outputs, name=self.name)


def attention_model(cf, name='attention_model'):
    input_shape = cf.input_shape + (3,) if cf.color_mode == 'rgb' else cf.input_shape + (1,)
    if cf.core_model == 'vgg16':
        weights = 'imagenet' if cf.imagenet_weights else None
        core_model = VGG16(input_shape=input_shape, include_top=False, weights=weights)
        att_layers = np.array(['block3_pool', 'block4_pool', 'block5_pool'])[cf.branches]
    elif cf.core_model == 'resnet50':
        weights = 'imagenet' if cf.imagenet_weights else None
        core_model = ResNet50(input_shape=input_shape, include_top=False, weights=weights)
        att_layers = np.array([core_model.layers[p].name for p in np.array([-96, -34, -2])[cf.branches]])
    elif cf.core_model == 'joseph_basic':
        core_model = JosephClass(input_shape)
        att_layers = np.array(['Pool2', 'Pool3', 'Pool4'])[cf.branches]
    elif cf.core_model == 'joseph_extended':
        core_model = JosephExt(input_shape)
        att_layers = np.array(['Pool2', 'Pool3', 'Pool4'])[cf.branches]
    else:
        raise ValueError('Invalid core model')
    return _BaseModel(name, core_model, att_layers, cf.nb_classes).get_model(cf.multi_loss, cf.early_fusion)
