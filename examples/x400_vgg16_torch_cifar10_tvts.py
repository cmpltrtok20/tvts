import os
import sys
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from PyCmpltrtok.common import sep
from PyCmpltrtok.common_np import uint8_to_flt_by_lut
from PyCmpltrtok.common_torch import torch_compile, torch_acc_top1, torch_acc_top2, torch_fit, torch_evaluate, torch_infer
from PyCmpltrtok.common_gpgpu import get_gpu_indexes_from_env
import tvts.tvts as tvts


###############################################################################################################
# VGG model related (start)
class ConvBnRelu(torch.nn.Module):
    """
    A "Conv - Batch Norm - ReLU" Unit
    """
    def __init__(self, in_ch, filters, ksize=(3, 3), strides=(1, 1), padding='same', **kwargs):
        super().__init__(**kwargs)
        if padding == 'same':
            if type(ksize) == tuple:
                pad = (ksize[0] // 2, ksize[1] // 2)
            else:
                pad = ksize // 2
        else:
            pad = padding
        self.conv = torch.nn.Conv2d(in_ch, filters, ksize, strides, pad, bias=False)
        self.bn = torch.nn.BatchNorm2d(filters)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
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


class VGG(torch.nn.Module):
    """
    https://arxiv.org/abs/1409.1556
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    by
    Karen Simonyan, Andrew Zisserman
    """
    def __init__(self, n_cls, input_shape=(3, 224, 224), conf=VggD16Conf, **kwargs):
        """

        :param n_cls: Number of classes.
        :param input_shape: Input shape tuple in format (C, H, W)
        :param conf: List of below elements: ( See VggD16Conf as an example )
            Positive integer for ConvBnRelu 3x3/1 with that many filters.
            Negative integer for that many neuron's fully connected layer.
            Positive fraction for that many proportion's dropout.
            Letter m for MaxPool/2.
            Letter f for Flatten.
        :param kwargs: Other args.
        """
        super().__init__(**kwargs)
        layers = []
        in_ch, in_h, in_w = input_shape
        for conf_el in conf:
            if 'm' == conf_el:
                layers.append(torch.nn.MaxPool2d(2, 2, 0))
                in_h //= 2
                in_w //= 2
            elif 'f' == conf_el:
                layers.append(torch.nn.Flatten())
                in_ch = in_h * in_w * in_ch
            elif isinstance(conf_el, int):
                if conf_el > 0:
                    filters = conf_el
                    layers.append(ConvBnRelu(in_ch, filters))
                    in_ch = filters
                elif conf_el == 0:
                    raise ValueError("Filter count of conv layer or neuron number of connected layer cannot be zero!")
                else:
                    n_neuron = abs(conf_el)
                    layers.append(torch.nn.Linear(in_ch, n_neuron))
                    in_ch = n_neuron
                    layers.append(torch.nn.ReLU())
            elif isinstance(conf_el, float):
                layers.append(torch.nn.Dropout(conf_el))
            else:
                raise ValueError("Invalid config element {} ".format(conf_el))
        layers.append(torch.nn.Linear(in_ch, n_cls))
        layers.append(torch.nn.LogSoftmax(dim=1))
        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.seq(x)
        return x
# VGG model related (end)
###############################################################################################################


if '__main__' == __name__:
    import matplotlib.pyplot as plt
    import argparse
    import PyCmpltrtok.data.cifar10.load_cifar10 as cifar10
    from PyCmpltrtok.auth.mongo.conn import conn as mconn

    def _main():
        """ The main program. """
        sep('VGG16 by PyTorch on Cifar10 with TVTS')

        ###############################################################################################################
        # Hyper params and switches (start)
        sep('Decide hyper params')
        VER = 'v3.0'  # version info of this code file
        MEMO = VER  # default memo
        N_BATCH_SIZE = 256
        N_EPOCHS = 4
        LR = 0.001  # default init learning rate
        GAMMA = 0.98  # default multiplicative factor of learning rate decay per epoch
        GAMMA_STRATEGY = 'step'
        GAMMA_STEP = 32
        WARMUP = 128
        print(f'Default LR={LR}, GAMMA={GAMMA}')
        IS_SPEC_LR = False  # is manually specified the LR
        IS_SPEC_GAMMA = False  # is manually specified the GAMMA

        # default dir for saving weights
        BASE_DIR, FILE_NAME = os.path.split(os.path.abspath(__file__))
        SAVE_DIR = os.path.join(BASE_DIR, '_save', FILE_NAME, VER)

        # specify or override params from CLI
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # group #1
        parser.add_argument('--name', help='The name of this training, VERY important to TVTS.', type=str, default='tvts_ex_vgg16_torch_cifar10')
        parser.add_argument('--memo', help='The memo.', type=str, default='(no memo)')
        parser.add_argument('--temp', help='Run as temporary code', action='store_true')
        parser.add_argument('-t', '--test', help='Only run testing phase, no training.', action='store_true')
        # group #2
        parser.add_argument('-n', '--epochs', help='How many epoches to train.', type=int, default=N_EPOCHS)
        parser.add_argument('--batch', help='Batch size.', type=int, default=N_BATCH_SIZE)
        parser.add_argument('--lr', help='Learning rate.', type=float, default=None)
        parser.add_argument('--warmup', help='Warmup step(s) or epoch(s)depending on --gamma-strategy.', type=int, default=WARMUP)
        parser.add_argument('--gamma', help='Multiplicative factor of learning rate decay per epoch.', type=float, default=None)
        parser.add_argument('--gamma_strategy', help='Step Gamma LR scheduler on step(s) or epoch(s)', type=str, default=GAMMA_STRATEGY)
        parser.add_argument('--gamma_step', help='Step Gamma LR scheduler per --gamma-step steps of step(s) or epoch(s) depending on --gamma-strategy.', type=int, default=GAMMA_STEP)
        # group #3
        parser.add_argument('--pi', help='id of the parent training', type=int, default=0)
        parser.add_argument('--pe', help='parent epoch of the parent training', type=int, default=0)
        parser.add_argument('--save_freq', help='How many epochs save weights once.', type=int, default=1)
        parser.add_argument('--save_dir', help='The dir where weights saved.', type=str, default=SAVE_DIR)
        # group #4
        parser.add_argument('--init_weights', help='The path to the stored weights to init the model.', type=str, default=None)
        parser.add_argument('--link', help='config name of the mongodb', type=str, default=tvts.DEFAULT_LINK)
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
        n_epochs = args.epochs
        assert n_epochs >= 0
        n_batch_size = args.batch
        assert n_batch_size > 0
        lr = args.lr
        gamma = args.gamma
        if lr is not None:
            assert lr > 0
            IS_SPEC_LR = True
        else:
            lr = LR
        if gamma is not None:
            assert 0 < gamma <= 1
            IS_SPEC_GAMMA = True
        else:
            gamma = GAMMA
        gamma_step = args.gamma_step
        gamma_strategy = args.gamma_strategy
        
        warmup = args.warmup
        
        IS_TRAIN = not not n_epochs
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
        link = args.link
        if link is not None:
            client = mconn(link)
            host = client.HOST
            port = client.PORT
        else:
            client = None
            link = None
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
            'batch_size': n_batch_size,
            'lr': lr,
            'gamma': gamma,
            'gamma_strategy': gamma_strategy,
            'gamma_step': gamma_step,
            'warmup': warmup,
            'n_epoch': n_epochs,
        }
        ts = tvts.Tvts(
            NAME,
            memo=MEMO,
            is_temp=TEMP,
            mongo_link=client,
            host=MONGODB_HOST, port=MONGODB_PORT,
            save_freq=SAVE_FREQ, save_dir=SAVE_DIR,
            init_weights=INIT_WEIGHTS,
            params=xparams
        )
        print(f'TVTS initialized. TRAIN ID = {ts.train_id}')
        print(f'MONGODB_HOST={MONGODB_HOST}, MONGODB_PORT={MONGODB_PORT}, NAME={NAME}, MEMO={MEMO}, TEMP={TEMP}')
        print(f'SAVE_FREQ={SAVE_FREQ}, SAVE_DIR={SAVE_DIR}')
        print(f'INIT_WEIGHTS={INIT_WEIGHTS}')
        for k in sorted(xparams.keys()):
            print(f"{k} = |{xparams[k]}|")
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
                warmup = 0
                if not IS_SPEC_LR:
                    lr = ts.params['lr']
                if not IS_SPEC_GAMMA:
                    gamma = ts.params['gamma']
                    gamma_strategy = ts.params['gamma_strategy']
                    gamma_step = ts.params['gamma_step']
                print(f'Restored: lr={lr}, GAMMA={GAMMA}')
        print(f'CKPT_PATH={CKPT_PATH}')
        print('Use below CLI command to visualize this training by TVTS:')
        if link is None:
            mongo_str = f'--host "{MONGODB_HOST}" --port {MONGODB_PORT}'
        else:
            mongo_str = f'--link "{link}"'
        print(f'python3 /path/to/tvts/tvts.py {mongo_str} -m "loss|loss_val,top1|top1_val|top2|top2_val" --batch_metrics "loss,top1|top2" -k "top1_val" --save_dir "{SAVE_DIR}" "{NAME}"')
        print('Allright? (y/[N]):', end='', flush=True)
        xinput = input().strip().lower()
        if 'y' != xinput:
            print("OK, let's stop it.")
            sys.exit(0)
        # tvts init and resume (end)
        ###############################################################################################################

        ###############################################################################################################
        # model and data (start)

        # select device
        sep('cpu or gpu')
        # device_id = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        n_gpus = torch.cuda.device_count()
        if not n_gpus:
            device_id = 'cpu'
        else:
            # gpu_id = n_gpus - 1
            gpu_id = 0
            device_id = f'cuda:{gpu_id}'
        device = torch.device(device_id)
        visible_gpus = get_gpu_indexes_from_env()
        print(device)
        if len(visible_gpus):
            print(f'Device #{visible_gpus[-1]}')

        # the model
        sep('The model')
        model = VGG(10, (3, 32, 32)).to(device)
        # print(model)
        model_dict = torch_compile(
            ts, device, model, torch.nn.NLLLoss(),
            torch.optim.Adam, ALPHA=lr, GAMMA=GAMMA, GAMMA_STRATEGY=gamma_strategy, GAMMA_STEP=gamma_step,
            warmup=warmup,
            metrics={
                'top1': torch_acc_top1,
                'top2': torch_acc_top2,
            },
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
        x_train = x_train.reshape(-1, *shape_)
        x_test = x_test.reshape(-1, *shape_)
        # to float
        x_train = uint8_to_flt_by_lut(x_train)
        x_test = uint8_to_flt_by_lut(x_test)
        # to tensor
        x_train = torch.Tensor(x_train)
        x_test = torch.Tensor(x_test)
        y_train = torch.Tensor(y_train)
        y_test = torch.Tensor(y_test)
        # to Dataset
        ds_test = TensorDataset(x_test, y_test)
        dl_test = DataLoader(ds_test, N_BATCH_SIZE, drop_last=False)

        # restore check point
        if CKPT_PATH is None:
            sep('From scratch')
        else:
            if not os.path.exists(CKPT_PATH):
                raise tvts.TvtsException(f'CKPT {CKPT_PATH} does not exist!')
            sep('Restore check point')
            print(f'Loading weight from {CKPT_PATH} ...')
            sdict = torch.load(CKPT_PATH)

            # example: use former weights after added a dropout layer
            if 0:
                w21 = sdict['seq.21.weight']
                b21 = sdict['seq.21.bias']
                w23 = sdict['seq.23.weight']
                b23 = sdict['seq.23.bias']
                del sdict['seq.21.weight']
                del sdict['seq.21.bias']
                del sdict['seq.23.weight']
                del sdict['seq.23.bias']
                sdict['seq.22.weight'] = w21
                sdict['seq.22.bias'] = b21
                sdict['seq.24.weight'] = w23
                sdict['seq.24.bias'] = b23

            model.load_state_dict(sdict)
            print('Loaded.')
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
            n_batches = int(np.floor(N_SAMPLE_AMOUNT / N_BATCH_SIZE))
            print('N_SAMPLE_AMOUNT:', N_SAMPLE_AMOUNT, file=sys.stderr)
            print('N_BATCH_SIZE:', N_BATCH_SIZE, file=sys.stderr)
            print('n_batches (per epoch):', n_batches, file=sys.stderr)
        
            ds_train = TensorDataset(x_train, y_train)
            ds_val = TensorDataset(x_test, y_test)
            dl_train = DataLoader(ds_train, N_BATCH_SIZE, drop_last=True, shuffle=True)
            dl_val = DataLoader(ds_val, N_BATCH_SIZE, drop_last=False)
            torch_fit(model_dict, dl_train, dl_val, n_epochs)
        # train (end)
        ###############################################################################################################

        ###############################################################################################################
        # test (start)
        sep('Test')
        print('Evaluating ...')
        avg_loss_test, avg_metric_test = torch_evaluate(model_dict, dl_test)
        print(f'avg_loss_test={avg_loss_test}, avg_metric_test={avg_metric_test}')
        # test (end)
        ###############################################################################################################

        ###############################################################################################################
        # demo (start)
        sep('Demo')
        spr = 4
        spc = 5
        spn = 0
        plt.figure(figsize=[8, 6])
        n_demo = spr * spc
        x_demo = x_test.numpy()[:n_demo]
        y_demo = y_test.numpy().astype(int)[:n_demo]
        h_demo = torch_infer(x_demo, model, device, batch_size=8).argmax(axis=1)
        for i in range(n_demo):
            spn += 1
            plt.subplot(spr, spc, spn)
            htype = label_names[h_demo[i]]
            gttype = label_names[y_demo[i]]
            right = True if htype == gttype else False
            title = gttype if right else f'{htype}(gt: {gttype})'
            plt.title(title, color="black" if right else "red")
            plt.axis('off')
            plt.imshow(np.transpose(x_test[i].reshape(*shape_), (1, 2, 0)))
        print('Check and close the plotting window to continue ...')
        plt.show()
        # demo (end)
        ###############################################################################################################

    _main()  # Main program entrance
    sep('All over')
