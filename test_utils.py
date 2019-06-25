import os

import keras.backend as K
import matplotlib as mpl
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

mpl.use('Agg')
import matplotlib.pyplot as plt


def get_layer(model, name):
    for layer in model.layers:
        if layer.name == name:
            return layer

def save_att_maps(model, sample, branches, output_path):
    sample = np.expand_dims(sample, axis=0)
    att_layers = [get_layer(model, name) for name in np.array(['Att_1', 'Att_2', 'Att_3'])[branches]]
    plt.imshow(sample[0, :, :, 0], cmap='bone')
    plt.savefig(os.path.join(output_path, 'img.png'))
    for idx, attn_layer in enumerate(att_layers):
        attn_func = K.function(inputs=[model.get_input_at(0), K.learning_phase()],
                               outputs=[attn_layer.get_output_at(0)])
        att_img = attn_func([sample, 0])[0]
        plt.imshow(att_img[0, :, :, 0], cmap='viridis', vmin=0, vmax=1, interpolation='lanczos')
        plt.savefig(os.path.join(output_path, 'attention_map_{0}.png'.format(str(idx))))


def save_curves(output_path, multi_loss=False, early_fusion=False, output_path_base=None):
    plt.close()
    event_acc = EventAccumulator(output_path)
    event_acc.Reload()
    tags = np.sort([s.split('_')[2] for s in event_acc.Tags()['scalars']
                    if 'val_dense' in s and '_loss' in s])
    if early_fusion:
        assert output_path_base is not None
        event_acc = EventAccumulator(output_path_base)
        event_ef_acc = EventAccumulator(output_path)
        event_acc.Reload()
        event_ef_acc.Reload()

        _, step_nums, train_acc = zip(*event_acc.Scalars('acc'))
        _, step_nums, val_acc = zip(*event_acc.Scalars('val_acc'))
        _, step_nums, train_loss = zip(*event_acc.Scalars('loss'))
        _, step_nums, val_loss = zip(*event_acc.Scalars('val_loss'))
        _, step_nums_ef, train_acc_ef = zip(*event_ef_acc.Scalars('acc'))
        _, step_nums_ef, val_acc_ef = zip(*event_ef_acc.Scalars('val_acc'))
        _, step_nums_ef, train_loss_ef = zip(*event_ef_acc.Scalars('loss'))
        _, step_nums_ef, val_loss_ef = zip(*event_ef_acc.Scalars('val_loss'))

        plt.plot(step_nums, train_loss, '--', color='#0485d1', label='loss')
        plt.plot(step_nums, val_loss, '--', color='#f97306', label='val_loss')
        plt.plot(step_nums_ef, train_loss_ef, color='#0485d1', label='loss_FC')
        plt.plot(step_nums_ef, val_loss_ef, color='#f97306', label='val_loss_FC')
        plt.title('Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        idx = np.argmin(val_loss)
        idx_ef = np.argmin(val_loss_ef)
        plt.plot(step_nums[idx], val_loss[idx], '*', color='g')
        plt.plot(step_nums_ef[idx_ef], val_loss_ef[idx_ef], '*', color='g')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(output_path, 'loss_FC.png'))
        plt.close()

        plt.plot(step_nums, train_acc, '--', color='#0485d1')
        plt.plot(step_nums, val_acc, '--', color='#f97306')
        plt.plot(step_nums_ef, train_acc_ef, color='#0485d1')
        plt.plot(step_nums_ef, val_acc_ef, color='#f97306')
        plt.title('Acc')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        idx = np.argmin(val_loss)
        idx_ef = np.argmin(val_loss_ef)
        plt.plot(step_nums[idx], val_acc[idx], '*', color='g')
        plt.plot(step_nums_ef[idx_ef], val_acc_ef[idx_ef], '*', color='g')
        plt.close()
        plt.savefig(os.path.join(output_path, 'acc_FC.png'))

    else:
        if not multi_loss:
            _, step_nums, train_acc = zip(*event_acc.Scalars('acc'))
            _, step_nums, val_acc = zip(*event_acc.Scalars('val_acc'))
            _, step_nums, train_loss = zip(*event_acc.Scalars('loss'))
            _, step_nums, val_loss = zip(*event_acc.Scalars('val_loss'))
            step_nums = np.array(step_nums)

            plt.plot(step_nums, train_loss)
            plt.plot(step_nums, val_loss)
            plt.title('Loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            idx = np.argmin(val_loss)
            plt.plot(step_nums[idx], val_loss[idx], '*')
            plt.savefig(os.path.join(output_path, 'loss.png'))
            plt.close()

            plt.plot(step_nums, train_acc)
            plt.plot(step_nums, val_acc)
            plt.title('Acc')
            plt.xlabel('epoch')
            plt.ylabel('acc')
            idx = np.argmin(val_loss)
            plt.plot(step_nums[idx], train_acc[idx], '*', color='g')
            plt.close()
            plt.savefig(os.path.join(output_path, 'acc.png'))

        else:
            _, step_nums, loss = zip(*event_acc.Scalars('loss'))
            _, step_nums, val_loss = zip(*event_acc.Scalars('val_loss'))
            _, step_nums, dense_1_loss = zip(*event_acc.Scalars('dense_' + tags[0] + '_loss'))
            _, step_nums, val_dense_1_loss = zip(*event_acc.Scalars('val_dense_' + tags[0] + '_loss'))
            _, step_nums, dense_2_loss = zip(*event_acc.Scalars('dense_' + tags[1] + '_loss'))
            _, step_nums, val_dense_2_loss = zip(*event_acc.Scalars('val_dense_' + tags[1] + '_loss'))
            _, step_nums, dense_1_acc = zip(*event_acc.Scalars('dense_' + tags[0] + '_acc'))
            _, step_nums, val_dense_1_acc = zip(*event_acc.Scalars('val_dense_' + tags[0] + '_acc'))
            _, step_nums, dense_2_acc = zip(*event_acc.Scalars('dense_' + tags[1] + '_acc'))
            _, step_nums, val_dense_2_acc = zip(*event_acc.Scalars('val_dense_' + tags[1] + '_acc'))
            step_nums = np.array(step_nums)

            plt.plot(step_nums, loss, color='#0485d1')
            plt.plot(step_nums, val_loss, color='#f97306')
            plt.title('Global Loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            idx = np.argmin(val_loss)
            plt.plot(step_nums[idx], val_loss[idx], '*', color='g')
            plt.savefig(os.path.join(output_path, 'global_loss.png'))
            plt.close()

            plt.plot(step_nums, dense_1_loss, color='#0485d1', label='Att0')
            plt.plot(step_nums, val_dense_1_loss, color='#f97306', label='val_Att0')
            plt.plot(step_nums[idx], val_dense_1_loss[idx], '*', color='g')
            plt.plot(step_nums, dense_2_loss, '--', color='#0485d1', label='Att1')
            plt.plot(step_nums, val_dense_2_loss, '--', color='#f97306', label='val_Att1')
            plt.plot(step_nums[idx], val_dense_2_loss[idx], '*', color='g')
            plt.title('Branches Loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(output_path, 'multi_loss.png'))
            plt.close()

            plt.plot(step_nums, dense_1_acc, color='#0485d1', label='Att0')
            plt.plot(step_nums, val_dense_1_acc, color='#f97306', label='val_Att0')
            plt.plot(step_nums[idx], val_dense_1_acc[idx], '*', color='g')
            plt.plot(step_nums, dense_2_acc, '--', color='#0485d1', label='Att1')
            plt.plot(step_nums, val_dense_2_acc, '--', color='#f97306', label='val_Att1')
            plt.plot(step_nums[idx], val_dense_2_acc[idx], '*', color='g')
            plt.title('Branches Acc')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc='upper right')
            plt.close()
            plt.savefig(os.path.join(output_path, 'multi_acc.png'))