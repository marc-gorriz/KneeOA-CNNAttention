import os

from keras.preprocessing.image import ImageDataGenerator


def preprocess_input(x):
    x = x[..., ::-1]
    mean = [120.67308000216417, 120.67308000216417, 120.67308000216417]
    std = [75.15477869837049, 75.15477869837049, 75.15477869837049]
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    if std is not None:
        x[..., 0] /= std[0]
        x[..., 1] /= std[1]
        x[..., 2] /= std[2]
    return x


def flow_multiloss(generator, nb):
    for x, y in generator:
        yield x, [y] * nb


def get_generators(cf):
    core_train = ImageDataGenerator(horizontal_flip=cf.horizontal_flip,
                                    preprocessing_function=preprocess_input)

    core_test = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = core_train.flow_from_directory(directory=os.path.join(cf.data_path, 'train'),
                                               target_size=cf.input_shape,
                                               color_mode=cf.color_mode,
                                               class_mode='categorical',
                                               batch_size=cf.batch_size,
                                               shuffle=cf.shuffle)

    val_gen = core_test.flow_from_directory(directory=os.path.join(cf.data_path, 'validation'),
                                            target_size=cf.input_shape,
                                            color_mode=cf.color_mode,
                                            class_mode='categorical',
                                            batch_size=cf.batch_size,
                                            shuffle=False)

    test_gen = core_test.flow_from_directory(directory=os.path.join(cf.data_path, 'test'),
                                             target_size=cf.input_shape,
                                             color_mode=cf.color_mode,
                                             class_mode='categorical',
                                             batch_size=cf.batch_size,
                                             shuffle=False)
    return train_gen, val_gen, test_gen
