from x400_vgg16_torch_cifar10_tvts import VGG
import os
import re
import sys
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from PyCmpltrtok.common import sep
from PyCmpltrtok.common_np import uint8_to_flt_by_lut
from PyCmpltrtok.common_torch import torch_compile, torch_acc_top1, torch_acc_top2, torch_fit, torch_evaluate, torch_infer
from PyCmpltrtok.common_gpgpu import get_gpu_indexes_from_env
import tvts.tvts as tvts


class VGGPara(VGG):
    
    def __init__(self, dev0, dev1, *args, **kwargs):
        super(VGGPara, self).__init__(*args, **kwargs)

        self.dev0 = dev0
        self.dev1 = dev1

        # indexed from 0 and incremental, but hard to trace the relationship with the vanilla model
        # self.convs = torch.nn.Sequential(*self.seq[:18])
        # self.fc1 = torch.nn.Sequential(*self.seq[18:22])
        # self.fc2 = torch.nn.Sequential(*self.seq[22:24])
        # self.fc3 = torch.nn.Sequential(*self.seq[24:])

        # indexed from not 0 and non-incremental, but easy to link to the vanilla model
        self.convs = self.seq[:18]
        self.fc1 = self.seq[18:22]
        self.fc2 = self.seq[22:24]
        self.fc3 = self.seq[24:]

        self.convs = self.convs.to(dev0)
        self.fc1 = self.fc1.to(dev0)
        self.fc2 = self.fc2.to(dev1)
        self.fc3 = self.fc3.to(dev1)

    def forward(self, x):
        # x = x.to(self.dev0)
        x = self.convs(x)
        x = self.fc1(x)

        x = x.to(self.dev1)
        x = self.fc2(x)
        x = self.fc3(x)

        x = x.to(self.dev0)
        return x

    @staticmethod
    def migrate_from_vanilla_vgg(sdict):
        """Missing key(s) in state_dict: 
        "convs.0.conv.weight", 
        "convs.0.bn.weight",
        "convs.0.bn.bias",
        "convs.0.bn.running_mean", 
        "convs.0.bn.running_var",
        "convs.1.conv.weight",
        "convs.1.bn.weight",
        "convs.1.bn.bias", 
        "convs.1.bn.running_mean",
        "convs.1.bn.running_var",
        "convs.3.conv.weight",
        "convs.3.bn.weight",
        "convs.3.bn.bias",
        "convs.3.bn.running_mean",
        "convs.3.bn.running_var",
        "convs.4.conv.weight",
        "convs.4.bn.weight",
        "convs.4.bn.bias",
        "convs.4.bn.running_mean",
        "convs.4.bn.running_var",
        "convs.6.conv.weight",
        "convs.6.bn.weight",
        "convs.6.bn.bias",
        "convs.6.bn.running_mean",
        "convs.6.bn.running_var",
        "convs.7.conv.weight",
        "convs.7.bn.weight",
        "convs.7.bn.bias",
        "convs.7.bn.running_mean",
        "convs.7.bn.running_var",
        "convs.8.conv.weight",
        "convs.8.bn.weight",
        "convs.8.bn.bias",
        "convs.8.bn.running_mean",
        "convs.8.bn.running_var",
        "convs.10.conv.weight",
        "convs.10.bn.weight",
        "convs.10.bn.bias",
        "convs.10.bn.running_mean",
        "convs.10.bn.running_var",
        "convs.11.conv.weight",
        "convs.11.bn.weight",
        "convs.11.bn.bias",
        "convs.11.bn.running_mean",
        "convs.11.bn.running_var",
        "convs.12.conv.weight",
        "convs.12.bn.weight",
        "convs.12.bn.bias",
        "convs.12.bn.running_mean",
        "convs.12.bn.running_var",
        "convs.14.conv.weight",
        "convs.14.bn.weight",
        "convs.14.bn.bias",
        "convs.14.bn.running_mean",
        "convs.14.bn.running_var",
        "convs.15.conv.weight",
        "convs.15.bn.weight",
        "convs.15.bn.bias",
        "convs.15.bn.running_mean",
        "convs.15.bn.running_var",
        "convs.16.conv.weight",
        "convs.16.bn.weight",
        "convs.16.bn.bias",
        "convs.16.bn.running_mean",
        "convs.16.bn.running_var",
        "fc1.1.weight",
        "fc1.1.bias",
        "fc2.0.weight",
        "fc2.0.bias",
        "fc3.0.weight",
        "fc3.0.bias". 
        """
        regexp = re.compile(r'^seq\.(\d+)\.conv\.weight$')
        keys = []
        for k in sdict.keys():
            keys.append(k)
        for k in keys:
            match = regexp.match(k)
            if not match:
                continue
            i = int(match[1])
            print(f'seq.{i}.conv.weight -> convs.{i}.conv.weight')
            sdict[f'convs.{i}.conv.weight'] = sdict[f'seq.{i}.conv.weight']
            for name in ['weight', 'bias', 'running_mean', 'running_var']:
                print(f'seq.{i}.bn.{name} -> convs.{i}.bn.{name}')
                sdict[f'convs.{i}.bn.{name}'] = sdict[f'seq.{i}.bn.{name}']
        xmap = {
            'fc1.19': 'seq.19',
            'fc2.22': 'seq.22',
            'fc3.24': 'seq.24',
        }
        for k, v in xmap.items():
            for name in ['weight', 'bias']:
                print(f'{v}.{name} -> {k}.{name}')
                sdict[f'{k}.{name}'] = sdict[f'{v}.{name}']
        return sdict

    @staticmethod
    def migrate_from_ddp_vgg(sdict):
        """
        Unexpected key(s) in state_dict:
        "module.seq.0.conv.weight",
        "module.seq.0.bn.weight",
        "module.seq.0.bn.bias",
        "module.seq.0.bn.running_mean",
        "module.seq.0.bn.running_var",
        "module.seq.0.bn.num_batches_tracked",
        "module.seq.1.conv.weight",
        "module.seq.1.bn.weight",
        "module.seq.1.bn.bias",
        "module.seq.1.bn.running_mean",
        "module.seq.1.bn.running_var",
        "module.seq.1.bn.num_batches_tracked",
        "module.seq.3.conv.weight",
        "module.seq.3.bn.weight",
        "module.seq.3.bn.bias",
        "module.seq.3.bn.running_mean",
        "module.seq.3.bn.running_var",
        "module.seq.3.bn.num_batches_tracked",
        "module.seq.4.conv.weight",
        "module.seq.4.bn.weight",
        "module.seq.4.bn.bias",
        "module.seq.4.bn.running_mean",
        "module.seq.4.bn.running_var",
        "module.seq.4.bn.num_batches_tracked",
        "module.seq.6.conv.weight",
        "module.seq.6.bn.weight",
        "module.seq.6.bn.bias",
        "module.seq.6.bn.running_mean",
        "module.seq.6.bn.running_var",
        "module.seq.6.bn.num_batches_tracked",
        "module.seq.7.conv.weight",
        "module.seq.7.bn.weight",
        "module.seq.7.bn.bias",
        "module.seq.7.bn.running_mean",
        "module.seq.7.bn.running_var",
        "module.seq.7.bn.num_batches_tracked",
        "module.seq.8.conv.weight",
        "module.seq.8.bn.weight",
        "module.seq.8.bn.bias",
        "module.seq.8.bn.running_mean",
        "module.seq.8.bn.running_var",
        "module.seq.8.bn.num_batches_tracked",
        "module.seq.10.conv.weight",
        "module.seq.10.bn.weight",
        "module.seq.10.bn.bias",
        "module.seq.10.bn.running_mean",
        "module.seq.10.bn.running_var",
        "module.seq.10.bn.num_batches_tracked",
        "module.seq.11.conv.weight",
        "module.seq.11.bn.weight",
        "module.seq.11.bn.bias",
        "module.seq.11.bn.running_mean",
        "module.seq.11.bn.running_var",
        "module.seq.11.bn.num_batches_tracked",
        "module.seq.12.conv.weight",
        "module.seq.12.bn.weight",
        "module.seq.12.bn.bias",
        "module.seq.12.bn.running_mean",
        "module.seq.12.bn.running_var",
        "module.seq.12.bn.num_batches_tracked",
        "module.seq.14.conv.weight",
        "module.seq.14.bn.weight",
        "module.seq.14.bn.bias",
        "module.seq.14.bn.running_mean",
        "module.seq.14.bn.running_var",
        "module.seq.14.bn.num_batches_tracked",
        "module.seq.15.conv.weight",
        "module.seq.15.bn.weight",
        "module.seq.15.bn.bias",
        "module.seq.15.bn.running_mean",
        "module.seq.15.bn.running_var",
        "module.seq.15.bn.num_batches_tracked",
        "module.seq.16.conv.weight",
        "module.seq.16.bn.weight",
        "module.seq.16.bn.bias",
        "module.seq.16.bn.running_mean",
        "module.seq.16.bn.running_var",
        "module.seq.16.bn.num_batches_tracked",
        "module.seq.19.weight",
        "module.seq.19.bias",
        "module.seq.22.weight",
        "module.seq.22.bias",
        "module.seq.24.weight",
        "module.seq.24.bias".
        """
        regexp = re.compile(r'^module\.(.+)$')
        keys = []
        for k in sdict.keys():
            keys.append(k)
        for k in keys:
            match = regexp.match(k)
            if not match:
                continue
            oldKey = match[0]
            thisKey = match[1]
            print(f'{oldKey} -> {thisKey}')
            sdict[thisKey] = sdict[oldKey]
            del sdict[oldKey]
        return sdict


if '__main__' == __name__:

    import matplotlib.pyplot as plt
    import argparse
    import PyCmpltrtok.data.cifar10.load_cifar10 as cifar10

    def _main():
        """ The main program. """
        sep('VGG16 by PyTorch on Cifar10 with TVTS')

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
        parser.add_argument('--name', help='The name of this training, VERY important to TVTS.', type=str,
                            default='tvts_ex_vgg16_torch_cifar10_03')
        parser.add_argument('--memo', help='The memo.', type=str, default='(no memo)')
        parser.add_argument('--temp', help='Run as temporary code', action='store_true')
        parser.add_argument('-t', '--test', help='Only run testing phase, no training.', action='store_true')
        # group #2
        parser.add_argument('-n', '--epochs', help='How many epoches to train.', type=int, default=2)
        parser.add_argument('--batch', help='Batch size.', type=int, default=256)
        parser.add_argument('--lr', help='Learning rate.', type=float, default=None)
        parser.add_argument('--gamma', help='Multiplicative factor of learning rate decay per epoch.', type=float,
                            default=None)
        # group #3
        parser.add_argument('--pi', help='id of the parent training', type=int, default=0)
        parser.add_argument('--pe', help='parent epoch of the parent training', type=int, default=0)
        parser.add_argument('--save_freq', help='How many epochs save weights once.', type=int, default=1)
        parser.add_argument('--save_dir', help='The dir where weights saved.', type=str, default=SAVE_DIR)
        # group #4
        parser.add_argument('--init_weights', help='The path to the stored weights to init the model.', type=str,
                            default=None)
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
            raise tvts.TvtsException(
                f'You cannot specify parent_id={PARENT_TRAIN_ID} and init_weights={INIT_WEIGHTS} at the same time!')
        # Hyper params and switches (end)
        ###############################################################################################################

        ###############################################################################################################
        # tvts init and resume (start)
        # tvts init
        sep('TVTS init')
        xparams = {
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
        print(
            f'python3 /path/to/tvts/tvts.py --host "{MONGODB_HOST}" --port {MONGODB_PORT} -m "loss|loss_val,top1|top1_val|top2|top2_val" --batch_metrics "loss,top1|top2" -k "top1_val" --save_dir "{SAVE_DIR}" "{NAME}"')
        # tvts init and resume (end)
        ###############################################################################################################

        ###############################################################################################################
        # model and data (start)

        # select device
        dev0 = torch.device('cuda:0')
        dev1 = torch.device('cuda:1')
        device = dev0

        # the model
        model = VGGPara(dev0, dev1, 10, (3, 32, 32))
        # print(model)
        # sys.exit(0)
        model_dict = torch_compile(
            ts, device, model, torch.nn.NLLLoss(),
            torch.optim.Adam, ALPHA=LR, GAMMA=GAMMA,
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

            try:
                model.load_state_dict(sdict)
            except:
                try:
                    print('Try to load as vanilla VGG model.')
                    sdict = VGGPara.migrate_from_vanilla_vgg(sdict)
                    model.load_state_dict(sdict)
                except:
                    print('Try to load as DDP VGG model.')
                    sdict = torch.load(CKPT_PATH)
                    sdict = VGGPara.migrate_from_ddp_vgg(sdict)
                    sdict = VGGPara.migrate_from_vanilla_vgg(sdict)
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
            torch_fit(model_dict, dl_train, dl_val, N_EPOCHS)
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
        plt.figure(figsize=[10, 8])
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
    