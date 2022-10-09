import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, losses, optimizers, metrics, activations
from PyCmpltrtok.common import sep
from PyCmpltrtok.common_np import uint8_to_flt_by_lut


###############################################################################################################
# VGG model related (start)
def ConvBnRelu(inputs, filters, ksize=(3, 3), strides=(1, 1), padding='same'):
    x = layers.Conv2D(filters, ksize, strides, padding, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


VggD16Conf = [
    64, 64, 'm',
    128, 128, 'm',
    256, 256, 256, 'm',
    512, 512, 512, 'm',
    512, 512, 512, 'm',
    # 'f', -1024, -1024,  # v1.0
    'f', -1024, 0.5, -1024,  # v2.0 with dropout
]  # It is almost a standard VGG-D-16 configuration except that I use FC-1024 x2 rather than FC-4096 x2.


def VGG(n_cls, input_shape=(224, 224, 3), conf=VggD16Conf):
    """
    https://arxiv.org/abs/1409.1556
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    by
    Karen Simonyan, Andrew Zisserman

    :param n_cls: Number of classes.
    :param input_shape: Input shape tuple in format (C, H, W)
    :param conf: List of below elements: ( See VggD16Conf as an example )
        Positive integer for ConvBnRelu 3x3/1 with that many filters.
        Negative integer for that many neuron's fully connected layer.
        Positive fraction for that many proportion's dropout.
        Letter m for MaxPool/2.
        Letter f for Flatten.
    :return:
    """
    inputs = keras.Input(input_shape)
    x = inputs
    for conf_el in conf:
        if 'm' == conf_el:
            x = layers.MaxPooling2D(strides=(2, 2), padding='same')(x)
        elif 'f' == conf_el:
            x = layers.Flatten()(x)
        elif isinstance(conf_el, int):
            if conf_el > 0:
                x = ConvBnRelu(x, conf_el)
            elif conf_el == 0:
                raise ValueError("Filter count of conv layer or neuron number of connected layer cannot be zero!")
            else:
                n_neuron = abs(conf_el)
                x = layers.Dense(n_neuron, activation=activations.relu)(x)
        elif isinstance(conf_el, float):
            x = layers.Dropout(conf_el)(x)
        else:
            raise ValueError("Invalid config element {} ".format(conf_el))
    x = layers.Dense(n_cls, activation=activations.softmax)(x)
    model = keras.Model(inputs, x)
    return model
# VGG model related (end)
###############################################################################################################


if '__main__' == __name__:
    import os
    import tvts.tvts as tvts
    import PyCmpltrtok.data.cifar10.load_cifar10 as cifar10
    import matplotlib.pyplot as plt
    import argparse

    def _main():
        """ The main program. """
        sep('VGG16 by Tensorflow 2 on Cifar10 with TVTS')

        ###############################################################################################################
        # Hyper params and switches (start)
        sep('Decide hyper params')
        VER = 'v3.0'  # version info of this code file
        MEMO = VER  # default memo
        LR = 0.001  # default init learning rate
        GAMMA = 0.95  # default multiplicative factor of learning rate decay per epoch
        print(f'Default LR={LR}, GAMMA={GAMMA}')
        IS_SPEC_LR = False  # is manually specified the LR
        IS_SPEC_GAMMA = False  # is manually specified the GAMMA

        # default dir for saving weights
        BASE_DIR, FILE_NAME = os.path.split(os.path.abspath(__file__))
        SAVE_DIR = os.path.join(BASE_DIR, '_save', FILE_NAME, VER)

        # specify or override params from CLI
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # group #1
        parser.add_argument('--name', help='The name of this training, VERY important to TVTS.', type=str, default='tvts_ex_vgg16_tf2_cifar10_02')
        parser.add_argument('--memo', help='The memo.', type=str, default='(no memo)')
        parser.add_argument('--temp', help='Run as temporary code', action='store_true')
        parser.add_argument('-t', '--test', help='Only run testing phase, no training.', action='store_true')
        # group #2
        parser.add_argument('-n', '--epochs', help='How many epoches to train.', type=int, default=2)
        parser.add_argument('--batch', help='Batch size.', type=int, default=256)
        parser.add_argument('--lr', help='Learning rate.', type=float, default=None)
        parser.add_argument('--gamma', help='Multiplicative factor of learning rate decay per epoch.', type=float, default=None)
        # group #3
        parser.add_argument('--pi', help='id of the parent training', type=int, default=0)
        parser.add_argument('--pe', help='parent epoch of the parent training', type=int, default=0)
        parser.add_argument('--save_freq', help='How many epochs save weights once.', type=int, default=1)
        parser.add_argument('--save_dir', help='The dir where weights saved.', type=str, default=SAVE_DIR)
        # group #4
        parser.add_argument('--init_weights', help='The path to the stored weights to init the model.', type=str, default=None)
        parser.add_argument('--host', help='Host of the mongodb for tvts.', type=str, default=tvts.DEFAULT_HOST)
        parser.add_argument('--port', help='Port of the mongodb for tvts.', type=str, default=tvts.DEFAULT_PORT)

        # parse CLI args
        args = parser.parse_args()
        # processing after parse
        # group #1
        NAME = args.name
        assert len(NAME) > 0
        memo = args.memo
        MEMO += '; ' + memo
        TEMP = args.temp
        if TEMP:
            MEMO = '(Temporary) ' + MEMO
        TEST = args.test
        # group #2
        N_EPOCHS = args.epochs
        assert N_EPOCHS >= 0
        N_BATCH_SIZE = args.batch
        assert N_BATCH_SIZE > 0
        lr = args.lr
        gamma = args.gamma
        if lr is not None:
            assert lr > 0
            LR = lr
            IS_SPEC_LR = True
        if gamma is not None:
            assert 0 < gamma <= 1
            GAMMA = gamma
            IS_SPEC_GAMMA = True
        IS_TRAIN = not not N_EPOCHS
        if TEST:
            IS_TRAIN = 0
        if not IS_TRAIN:
            sep('No training, just testing and demonstrating', char='<', rchar='>')
        # group #3
        PARENT_TRAIN_ID = args.pi
        PARENT_EPOCH = args.pe
        SAVE_FREQ = args.save_freq
        assert SAVE_FREQ > 0
        SAVE_DIR = args.save_dir
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f'Weights will be saved under this dir: "{SAVE_DIR}"')
        # group #4
        INIT_WEIGHTS = args.init_weights
        MONGODB_HOST = args.host
        MONGODB_PORT = args.port
        # init weights or parent training is mutual exclusion with each other
        if PARENT_TRAIN_ID and INIT_WEIGHTS:
            raise tvts.TvtsException(f'You cannot specify parent_id={PARENT_TRAIN_ID} and init_weights={INIT_WEIGHTS} at the same time!')
        # Hyper params and switches (end)
        ###############################################################################################################

        ###############################################################################################################
        # tvts init and resume (start)
        # tvts init
        sep('TVTS init')
        xparams={
            'ver': VER,
            'batch_size': N_BATCH_SIZE,
            'lr': LR,
            'gamma': GAMMA,
            'n_epoch': N_EPOCHS,
        }
        ts = tvts.Tvts(
            NAME,
            memo=MEMO,
            is_temp=TEMP,
            host=MONGODB_HOST, port=MONGODB_PORT,
            save_freq=SAVE_FREQ, save_dir=SAVE_DIR,
            init_weights=INIT_WEIGHTS,
            params=xparams
        )
        print(f'TVTS initialized. TRAIN ID = {ts.train_id}')
        print(f'MONGODB_HOST={MONGODB_HOST}, MONGODB_PORT={MONGODB_PORT}, NAME={NAME}, MEMO={MEMO}, TEMP={TEMP}')
        print(f'SAVE_FREQ={SAVE_FREQ}, SAVE_DIR={SAVE_DIR}')
        print(f'INIT_WEIGHTS={INIT_WEIGHTS}')
        print(xparams)
        # tvts resume
        CKPT_PATH = None
        if INIT_WEIGHTS:
            CKPT_PATH = INIT_WEIGHTS
        else:
            if PARENT_TRAIN_ID:
                sep('TVTS resume')
                print(f'PARENT_TRAIN_ID={PARENT_TRAIN_ID}, PARENT_EPOCH={PARENT_EPOCH}')
                rel_path, _ = ts.resume(PARENT_TRAIN_ID, PARENT_EPOCH, keys_of_data_to_restore=['lr', 'gamma'])
                CKPT_PATH = os.path.join(SAVE_DIR, rel_path)
                if not IS_SPEC_LR:
                    LR = ts.params['lr']
                if not IS_SPEC_GAMMA:
                    GAMMA = ts.params['gamma']
                print(f'Restored: LR={LR}, GAMMA={GAMMA}')
        print(f'CKPT_PATH={CKPT_PATH}')
        print('Use below CLI command to visualize this training by TVTS:')
        print(f'python3 /path/to/tvts/tvts.py --host "{MONGODB_HOST}" --port {MONGODB_PORT} -m "loss|loss_val,acc|acc_val" --batch_metrics "loss,acc" -k "acc_val" --save_dir "{SAVE_DIR}" "{NAME}"')
        # tvts init and resume (end)
        ###############################################################################################################

        ###############################################################################################################
        # model and data (start)


        # the model
        sep('The model')
        if CKPT_PATH is not None:
            if not os.path.exists(CKPT_PATH):
                raise Exception(f'CKPT {CKPT_PATH} does not exist!')
            sep('Resume')
            print(f'Loading model from {CKPT_PATH} ...')
            model = keras.models.load_model(CKPT_PATH, compile=False)
            print('Loaded.')
        else:
            sep('From scratch')
            model = VGG(10, (32, 32, 3))
        # model.summary()
        model.compile(
            optimizer=optimizers.Adam(learning_rate=LR),
            loss=losses.sparse_categorical_crossentropy,
            metrics=[metrics.sparse_categorical_accuracy],
        )

        # load data
        sep('The data')
        x_train, y_train, x_test, y_test, label_names = cifar10.load()
        shape_ = cifar10.shape_
        N_SAMPLE_AMOUNT = len(x_train)
        print('label_names', label_names)
        if TEMP:
            # This is the effect of the "temporary" code switch,
            # i.e. run on a small amount of data to just ensure the code is good.
            N_SAMPLE_AMOUNT = 1024
            x_train = x_train[:N_SAMPLE_AMOUNT]
            y_train = y_train[:N_SAMPLE_AMOUNT]
            x_test = x_test[:N_SAMPLE_AMOUNT]
            y_test = y_test[:N_SAMPLE_AMOUNT]
        # reshape

        x_train = x_train.reshape(-1, *shape_).transpose((0, 2, 3, 1))
        x_test = x_test.reshape(-1, *shape_).transpose((0, 2, 3, 1))
        # to float
        x_train = uint8_to_flt_by_lut(x_train)
        x_test = uint8_to_flt_by_lut(x_test)


        dl_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
            .shuffle(buffer_size=1024) \
            .batch(batch_size=N_BATCH_SIZE) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # model and data (end)
        ###############################################################################################################

        ###############################################################################################################
        # train (start)
        if not IS_TRAIN:
            sep('(Only testing, no training)', char='<', rchar='>')
        else:
            sep('Train')
            if TEMP:
                sep('(Temporary)', char='<', rchar='>')
            n_batches = int(np.ceil(N_SAMPLE_AMOUNT / N_BATCH_SIZE))
            print('N_SAMPLE_AMOUNT:', N_SAMPLE_AMOUNT, file=sys.stderr)
            print('N_BATCH_SIZE:', N_BATCH_SIZE, file=sys.stderr)
            print('n_batches (per epoch):', n_batches, file=sys.stderr)

            dl_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
                .shuffle(buffer_size=1024) \
                .batch(batch_size=N_BATCH_SIZE) \
                .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            dl_val = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
                .shuffle(buffer_size=1024) \
                .batch(batch_size=N_BATCH_SIZE) \
                .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            class LearningRateReducerCallback(keras.callbacks.Callback):

                def on_epoch_end(self, epoch, logs={}):
                    epoch = epoch + 1
                    old_lr = self.model.optimizer.lr.read_value()
                    new_lr = old_lr * GAMMA
                    print("\nEpoch: {}. Reducing Learning Rate from {} to {}\n".format(epoch, old_lr, new_lr))
                    self.model.optimizer.lr.assign(new_lr)

            class TvtsCallBack(keras.callbacks.Callback):

                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.epoch = 1

                def on_epoch_end(self, epoch, logs=None):
                    epoch = epoch + 1

                    if epoch and epoch % ts.save_freq == 0 or epoch == N_EPOCHS - 1:
                        # Warning: complex!
                        # Tensorflow will generate save_prefix.index and save_prefix.data-00000-of-00001 etc.
                        # TVTS need the path save_prefix.index to check the existence of the weights files.
						# It is your burden to make sure all files needed are there when loading the model.
                        save_name = f'{ts.get_save_name(epoch)}.h5'
                        save_path = os.path.join(SAVE_DIR, save_name)
                        print(f'Save modle to {save_path} ...')
                        model.save(save_path, include_optimizer=False, save_format='h5')
                        print('Saved.')
                        save_rel_path = os.path.relpath(save_path, SAVE_DIR)
                    else:
                        save_rel_path = None

                    lr = self.model.optimizer.lr.numpy().item()
                    ts.save_epoch(epoch, {
                        'loss': logs['loss'],
                        'acc': logs['sparse_categorical_accuracy'],
                        'loss_val': logs['val_loss'],
                        'acc_val': logs['val_sparse_categorical_accuracy'],
                        'lr': lr,
                    }, save_rel_path, ts.save_dir)
                    self.epoch += 1

                def on_train_batch_end(self, batch, logs=None):
                    batch = batch + 1

                    lr = self.model.optimizer.lr.numpy().item()
                    ts.save_batch(self.epoch, batch, {
                        'loss': logs['loss'],
                        'acc': logs['sparse_categorical_accuracy'],
                        'lr': lr,
                    })

            tvtsCallback = TvtsCallBack()
            lrCallback = LearningRateReducerCallback()
            model.fit(
                dl_train,
                epochs=N_EPOCHS,
                validation_data=dl_val,
                callbacks=[tvtsCallback, lrCallback],
            )
        # train (end)
        ###############################################################################################################

        ###############################################################################################################
        # test (start)
        sep('Test')
        print('Evaluating ...')
        model.evaluate(dl_test)
        # test (end)
        ###############################################################################################################

        ###############################################################################################################
        # demo (start)
        sep('Demo')
        spr = 4
        spc = 5
        spn = 0
        plt.figure(figsize=[10, 8])
        n_demo = spr * spc
        x_demo = x_test[:n_demo]
        y_demo = y_test[:n_demo]
        h_demo = model.predict(x_demo).argmax(axis=1)
        for i in range(n_demo):
            spn += 1
            plt.subplot(spr, spc, spn)
            htype = label_names[h_demo[i]]
            rtype = label_names[y_demo[i]]
            right = True if htype == rtype else False
            title = f'{rtype}=>{htype} ({"Y" if right else "X"})'
            plt.title(title, color="black" if right else "red")
            plt.axis('off')
            plt.imshow(x_test[i])
        print('Check and close the plotting window to continue ...')
        plt.show()
        # demo (end)
        ###############################################################################################################

    _main()  # Main program entrance
    sep('All over')
