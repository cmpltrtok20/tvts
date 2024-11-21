if '__main__' == __name__:
    import sys
    import numpy as np
    from PyCmpltrtok.common import sep
    from PyCmpltrtok.auth.mongo.conn import conn as mconn
    import tvts.tvts as tvts

    def _main():
        """ The main program. """
        sep('A fake simple demo of TVTS')

        ###############################################################################################################
        # Hyper params (start)
        sep('Decide hyper params')
        NAME = 'tvts_fake_demo_simple'
        OLD_TRAIN_ID = 4
        RESUME_EPOCH_BATCH = (5, 70)
        print('Use below CLI command to visualize this training by TVTS:')
        print(f'python3 tvts.py --link local -m "loss|loss_val,top1|top1_val" --batch_metrics "loss,top1" -k "top1_val" "{NAME}"')
        LR = 0.002  # default init learning rate
        GAMMA = 0.99  # default multiplicative factor of learning rate decay per epoch
        N_BATCH_SIZE = 256
        N_EPOCHS = 10
        N_SAMPLE_AMOUNT = 50000
        # Hyper params (end)
        ###############################################################################################################

        ###############################################################################################################
        # tvts init (start)
        sep('TVTS init')
        xparams={
            'batch_size': N_BATCH_SIZE,
            'lr': LR,
            'gamma': GAMMA,
            'n_epoch': N_EPOCHS,
            # 'train_id': 1,  # Example: Uncomment his line will trigger TvtsException, because of that "train_id" is a reserved key word.
        }
        client = mconn('local')
        ts = tvts.Tvts(
            NAME,
            old_train_id=OLD_TRAIN_ID,
            memo='The memo about this time of training',
            is_temp=True,
            mongo_link=client,
            db='tvts',
            table_prefix='train_log',
            params=xparams,
            save_freq=2,
            save_dir='/path/to/dir/for/saving/weights',  # In this fake demo, we do not actually save weights; but in practise it is a must.
        )
        print(f'TVTS initialized. TRAIN ID = {ts.train_id}')
        # tvts init and (end)
        ###############################################################################################################

        ###############################################################################################################
        # model and data (start)
        sep('model and data')
        # Actually no model and data, we just assume that there are N_SAMPLE_AMOUNT training data and the loss will decrease and
        # the top1 metric will increase by each mini-batch.
        LOSS_INIT, LOSS_VAL_INIT = 4.0, 5.5

        LOSS, LOSS_VAL = LOSS_INIT, LOSS_VAL_INIT
        TOP1, TOP1_VAL = 0, 0
        # model and data (end)
        ###############################################################################################################

        ###############################################################################################################
        # train (start)
        sep('Train')
        n_batches = int(np.ceil(N_SAMPLE_AMOUNT / N_BATCH_SIZE))
        print('N_SAMPLE_AMOUNT:', N_SAMPLE_AMOUNT, file=sys.stderr)
        print('N_BATCH_SIZE:', N_BATCH_SIZE, file=sys.stderr)
        print('n_batches (per epoch):', n_batches, file=sys.stderr)
        for epoch in range(N_EPOCHS):
            epoch += 1
            sep(epoch)
            
            if epoch < RESUME_EPOCH_BATCH[0]:
                continue
            
            for batch in range(n_batches):
                batch += 1

                if (epoch, batch) < RESUME_EPOCH_BATCH:
                    continue

                # We assume below is the effect of the training in one mini-batch
                LOSS *= 0.9988
                LOSS_VAL *= 0.9997
                TOP1, TOP1_VAL = (LOSS_INIT - LOSS) / LOSS_INIT, (LOSS_VAL_INIT - LOSS_VAL) / LOSS_VAL_INIT

                # Use tvts to save info of an mini-batch
                ts.save_batch(epoch, batch, {
                    'lr': LR,
                    'gamma': GAMMA,
                    'loss': LOSS,
                    'top1': TOP1,
                    # 'batch': 1,  # Example: Uncomment his line will trigger TvtsException, because of that "batch" is a reserved key word.
                })

                # the progress bar
                print('>', end='', flush=True)
                if not batch % 50:
                    print()

            # one epoch is over
            print()
            print(f'loss={LOSS}, loss_val={LOSS_VAL}, top1={TOP1}, top1_val={TOP1_VAL}')

            # Use tvts to save info of an epoch
            ts.save_epoch(epoch, {
                'lr': LR,
                'gamma': GAMMA,
                'loss': LOSS,
                'top1': TOP1,
                'loss_val': LOSS_VAL,
                'top1_val': TOP1_VAL,
                # 'epoch': 1,  # Example: Uncomment his line will trigger TvtsException, because of that "epoch" is a reserved key word.
            })
            
            ts.save_epoch(epoch, save_rel_path=f'checkpoint-{epoch}.sav', save_dir='/tmp/x150_fake_demo_simple_with_old_train_id.py/')
            
            # decay LR
            LR *= GAMMA

        # train (end)
        ###############################################################################################################

    _main()  # Main program entrance
    sep('All over')
