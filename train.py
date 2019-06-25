import argparse
import os
from importlib.machinery import SourceFileLoader

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam

from data import get_generators, flow_multiloss
from models.attention_model import attention_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train attention models')
    parser.add_argument('-c', '--config', type=str, default=None, help='config file')
    parser.add_argument('-g', '--gpu', type=str, default=0, help='gpu number')
    args = parser.parse_args()

    assert args.config is not None
    cf = SourceFileLoader('config', args.config).load_module()
    cf.output_path = os.path.join(cf.experiments_path, cf.experiment_name)
    if not os.path.exists(cf.experiments_path): os.makedirs(cf.experiments_path)
    if not os.path.exists(cf.output_path): os.makedirs(cf.output_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    train_gen, val_gen, _ = get_generators(cf)
    model = attention_model(cf)

    weight_path = os.path.join(cf.output_path, "weights.hdf5")
    optimizer = Adam(cf.lr, cf.beta)

    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=cf.plateau_factor,
                                  patience=cf.plateau_patience, min_lr=cf.plateau_min_lr)
    early_stop = EarlyStopping(monitor='val_loss', mode="min", patience=3)
    tensorboard = TensorBoard(log_dir=cf.output_path)
    callbacks_list = [checkpoint, reduce_lr, early_stop, tensorboard]

    if cf.multi_loss:
        model.compile(optimizer=optimizer, loss=['categorical_crossentropy'] * len(cf.branches),
                      loss_weights=cf.loss_weights, metrics=['accuracy'])
        model.fit_generator(flow_multiloss(train_gen, cf.branches),
                            validation_data=flow_multiloss(val_gen, cf.branches),
                            epochs=cf.epochs, steps_per_epoch=(train_gen.samples // cf.batch_size) * cf.data_rate,
                            validation_steps=val_gen.samples // cf.batch_size, callbacks=callbacks_list)
    else:
        model.compile(optimizer=optimizer, loss=['categorical_crossentropy'], metrics=['accuracy'])
        model.fit_generator(train_gen, val_gen, epochs=cf.epochs,
                            steps_per_epoch=(train_gen.samples // cf.batch_size) * cf.data_rate,
                            validation_steps=val_gen.samples // cf.batch_size, callbacks=callbacks_list)
