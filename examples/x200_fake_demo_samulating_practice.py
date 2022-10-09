if '__main__' == __name__:
    import sys
    import os
    import argparse
    import numpy as np
    from PyCmpltrtok.common import sep, get_path_from_prefix
    import tvts.tvts as tvts

    def _main():
        """ The main program. """
        sep('A fake demo of TVTS simulating real practice')

        ###############################################################################################################
        # Hyper params and switches (start)
        sep('Decide hyper params')
        VER = 'v1.0'  # version info of this code file
        MEMO = VER  # default memo is the version info
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
        parser.add_argument('--name', help='The name of this training, VERY important to TVTS.', type=str, default='tvts_py_example_x200_01')
        parser.add_argument('--memo', help='The memo.', type=str, default='(no memo)')
        parser.add_argument('--temp', help='Run as temporary code', action='store_true')
        # group #2
        parser.add_argument('-n', '--epochs', help='How many epoches to train.', type=int, default=10)
        parser.add_argument('--batch', help='Batch size.', type=int, default=256)
        parser.add_argument('--lr', help='Learning rate.', type=float, default=None)
        parser.add_argument('--gamma', help='Multiplicative factor of learning rate decay per epoch.', type=float, default=None)
        # group #3
        parser.add_argument('--pi', help='id of the parent training', type=int, default=0)
        parser.add_argument('--pe', help='parent epoch of the parent training', type=int, default=0)
        parser.add_argument('--save_freq', help='How many epochs save weights once.', type=int, default=2)
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
        # group #2
        N_EPOCHS = args.epochs
        assert N_EPOCHS > 0
        N_BATCH_SIZE = args.batch
        assert N_BATCH_SIZE > 0
        lr = args.lr
        gamma = args.gamma
        if lr is not None:
            assert lr > 0
            LR = lr
            IS_SPEC_LR = True
        if gamma is not None:
            assert 0 < gamma < 1
            GAMMA = gamma
            IS_SPEC_GAMMA = True
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
            # Note: Implement the semantic of INIT_WEIGHTS is your burden.
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
        print(f'python3 /path/to/tvts/tvts.py --host "{MONGODB_HOST}" --port {MONGODB_PORT} -m "loss|loss_val,top1|top1_val" --batch_metrics "loss,top1" -k "top1_val" --save_dir "{SAVE_DIR}" "{NAME}"')
        # tvts init and resume (end)
        ###############################################################################################################

        ###############################################################################################################
        # model and data (start)
        # Actually there is no model and data, we just assume that there are N_SAMPLE_AMOUNT training data and the loss will decrease and the top1 metric will increase by each mini-batch.
        if not TEMP:
            N_SAMPLE_AMOUNT = 50000
        else:
            N_SAMPLE_AMOUNT = 1024

        LOSS_INIT, LOSS_VAL_INIT, EPS = 5.0, 6.5, 1e-8

        def xload(xpath):
            loss, top1, loss_val, top1_val = np.load(xpath)
            return loss, top1, loss_val, top1_val

        def xsave(xpath, loss, top1, loss_val, top1_val):
            xarr = np.array([loss, top1, loss_val, top1_val], dtype=np.float32)
            np.save(xpath, xarr)

        def xtop1():
            return (LOSS_INIT - LOSS + EPS) / LOSS_INIT

        def xtop1_val():
            return (LOSS_VAL_INIT - LOSS_VAL + EPS) / LOSS_VAL_INIT

        # restore check point
        if CKPT_PATH is None:
            sep('From scratch')
            LOSS, LOSS_VAL = LOSS_INIT, LOSS_VAL_INIT
            TOP1, TOP1_VAL = xtop1(), xtop1_val()
        else:
            if not os.path.exists(CKPT_PATH):
                raise tvts.TvtsException(f'CKPT {CKPT_PATH} does not exist!')
            sep('Restore check point')
            print(f'Loading weight from {CKPT_PATH} ...')
            LOSS, TOP1, LOSS_VAL, TOP1_VAL = xload(CKPT_PATH)
            print('Loaded.')
        print(f'loss={LOSS}, loss_val={LOSS_VAL}, top1={TOP1}, top1_val={TOP1_VAL}')
        # model and data (end)
        ###############################################################################################################

        ###############################################################################################################
        # train (start)
        sep('Train')
        if TEMP:
            sep('(Temporary)', char='<', rchar='>')
        n_batches = int(np.ceil(N_SAMPLE_AMOUNT / N_BATCH_SIZE))
        print('N_SAMPLE_AMOUNT:', N_SAMPLE_AMOUNT, file=sys.stderr)
        print('N_BATCH_SIZE:', N_BATCH_SIZE, file=sys.stderr)
        print('n_batches (per epoch):', n_batches, file=sys.stderr)
        for epoch in range(N_EPOCHS):
            epoch += 1
            sep(epoch)
            for batch in range(n_batches):
                batch += 1

                # We assume below is the effect of the training in one mini-batch
                LOSS *= 0.9988
                LOSS_VAL *= 0.9997
                TOP1, TOP1_VAL = xtop1(), xtop1_val()

                # Note: use tvts to save info of an mini-batch
                ts.save_batch(epoch, batch, {
                    'lr': LR,
                    'gamma': GAMMA,
                    'loss': LOSS,
                    'top1': TOP1,
                })

                # the progress bar
                print('>', end='', flush=True)
                if not batch % 50:
                    print()

            # one epoch is over
            print()
            print(f'loss={LOSS}, loss_val={LOSS_VAL}, top1={TOP1}, top1_val={TOP1_VAL}')
            # save the last epoch or by ts.save_freq
            # Note: Implement the semantic of save_freq and save_rel_path is your burden.
            xargs = {}
            if epoch == N_EPOCHS or 0 == epoch % ts.save_freq:
                save_prefix = ts.get_save_name(epoch)
                save_rel_path = get_path_from_prefix(save_prefix, '.npy')
                # actually do the saving task
                save_path = os.path.join(ts.save_dir, save_rel_path)
                xsave(save_path, LOSS, TOP1, LOSS_VAL, TOP1_VAL)
                print(f'Saved to: {save_path}')
                xargs = dict(save_rel_path=save_rel_path, save_dir=ts.save_dir)
            # Note: use tvts to save info of an epoch
            ts.save_epoch(epoch, {
                'lr': LR,
                'gamma': GAMMA,
                'loss': LOSS,
                'top1': TOP1,
                'loss_val': LOSS_VAL,
                'top1_val': TOP1_VAL,
            }, **xargs)
            # decay LR
            LR *= GAMMA

        # train (end)
        ###############################################################################################################

    _main()  # Main program entrance
    sep('All over')
